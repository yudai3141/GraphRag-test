"""恐怖構造グラフを Neo4j 非依存でビルドし、スナップショット(JSON)に保存する。

memory/builder.py（Neo4j 版）の構築ロジックを忠実にミラーした in-memory 版。
実験の再現性のため、全グラフ（PTSD / 健常）を同一パイプラインでこのスクリプトから作る。

実行: uv run python -m caps5_graph_structure_eval_v3.build_snapshot <入力txt[,txt2,...]> <出力json>
  例: uv run python -m caps5_graph_structure_eval_v3.build_snapshot simulation/ptsd_story.txt caps5_graph_structure_eval_v3/data/graph_exp_ptsd.json
"""

import sys

import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import TokenTextSplitter

from actr_foa_kozak_v2 import config
from actr_foa_kozak_v2.domain.models import Edge, Episode, FearGraph, FearNode, node_key
from actr_foa_kozak_v2.infrastructure.embeddings import EmbeddingService
from actr_foa_kozak_v2.memory.extractor import FearStructureExtractor

from . import snapshot


def build_graph(source_paths: list[str]) -> FearGraph:
    config.load_env()
    extractor = FearStructureExtractor()
    embedder = EmbeddingService()

    docs = []
    for p in source_paths:
        docs += TextLoader(p).load()
    chunks = TokenTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    ).split_documents(docs)
    print(f"📄 {source_paths} を {len(chunks)} チャンクに分割")

    nodes: dict[str, FearNode] = {}          # key -> FearNode
    episodes: list[Episode] = []
    edge_w: dict[tuple, float] = {}          # (src,rel,dst) -> weight（構造は加算）
    stim_emb: dict[str, list] = {}
    order = 0
    n_frag = 0

    def add_node(label, name, embedding=None):
        key = node_key(label, name)
        if key not in nodes:
            nodes[key] = FearNode(key=key, label=label, name=name, embedding=embedding or [])
        elif embedding:
            nodes[key].embedding = embedding

    def add_edge(src_key, rel, dst_key, w=1.0, accumulate=True):
        k = (src_key, rel, dst_key)
        if accumulate:
            edge_w[k] = edge_w.get(k, 0.0) + w
        else:
            edge_w[k] = w

    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}/{len(chunks)}] 抽出中...")
        for frag in extractor.extract(chunk.page_content):
            n_frag += 1
            trigger = frag.trigger.strip()
            if not trigger:
                continue
            vec = embedder.embed(trigger).tolist()
            stim_emb[trigger] = vec
            add_node(config.STIMULUS_LABEL, trigger, vec)

            for cue in frag.other_cues:
                cue = cue.strip()
                if not cue or cue == trigger:
                    continue
                cvec = embedder.embed(cue).tolist()
                stim_emb[cue] = cvec
                add_node(config.STIMULUS_LABEL, cue, cvec)
                lo, hi = sorted([trigger, cue])
                add_edge(node_key(config.STIMULUS_LABEL, lo), "CO_OCCURS",
                         node_key(config.STIMULUS_LABEL, hi))

            resp_names = set()
            for resp in frag.responses:
                resp = resp.strip()
                if not resp:
                    continue
                resp_names.add(resp)
                add_node(config.RESPONSE_LABEL, resp)
                add_edge(node_key(config.STIMULUS_LABEL, trigger), "EVOKES",
                         node_key(config.RESPONSE_LABEL, resp))

            meaning_names = []
            for m in frag.meanings:
                text = m.text.strip()
                if not text:
                    continue
                meaning_names.append(text)
                add_node(config.MEANING_LABEL, text)
                if m.source in resp_names:
                    add_edge(node_key(config.RESPONSE_LABEL, m.source), "MEANS",
                             node_key(config.MEANING_LABEL, text))
                else:
                    add_edge(node_key(config.STIMULUS_LABEL, trigger), "MEANS",
                             node_key(config.MEANING_LABEL, text))

            if frag.episode is not None:
                order += 1
                ep = Episode(
                    id=f"fk_ep_{order:03d}",
                    context=frag.episode.context, event=frag.episode.event,
                    sensory=frag.episode.sensory,
                    valence=(frag.episode.valence or "negative").strip().lower(),
                    t_created=float(order),
                )
                ep.embedding = embedder.embed(ep.as_text()).tolist()
                episodes.append(ep)
                add_edge(node_key(config.STIMULUS_LABEL, trigger), "RECALLS", ep.key)
                for rname in resp_names:
                    add_edge(ep.key, "BINDS", node_key(config.RESPONSE_LABEL, rname))
                for mname in meaning_names:
                    add_edge(ep.key, "BINDS", node_key(config.MEANING_LABEL, mname))

    # 刺激般化 SIMILAR（重み=コサイン類似度・SET）
    names = list(stim_emb)
    if len(names) >= 2:
        M = np.asarray([stim_emb[n] for n in names], dtype=np.float32)
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        sims = M @ M.T
        for i, a in enumerate(names):
            cnt = 0
            for j in sims[i].argsort()[::-1]:
                if j == i:
                    continue
                sscore = float(sims[i, j])
                if sscore < config.STIM_SIMILAR_MIN or cnt >= config.STIM_SIMILAR_TOPN:
                    break
                lo, hi = sorted([a, names[j]])
                add_edge(node_key(config.STIMULUS_LABEL, lo), "SIMILAR",
                         node_key(config.STIMULUS_LABEL, hi), w=sscore, accumulate=False)
                cnt += 1

    edges = [Edge(src=s, dst=d, rel=r, weight=w) for (s, r, d), w in edge_w.items()]
    graph = FearGraph(nodes=list(nodes.values()), episodes=episodes, edges=edges)
    print(f"✅ fragment {n_frag} / ノード {len(graph.nodes)} / エピソード {len(episodes)} / エッジ {len(edges)}")
    return graph


def main() -> None:
    srcs = sys.argv[1].split(",")
    out = sys.argv[2]
    graph = build_graph(srcs)
    snapshot.dump(graph, out)
    # 感情価バランス
    from collections import Counter
    vc = Counter(e.valence for e in graph.episodes)
    print(f"   valence: {dict(vc)}  → {out}")


if __name__ == "__main__":
    main()
