"""恐怖構造グラフを構築して Neo4j に登録する（CLI）。

実行: uv run python -m actr_foa_kozak_v2.memory.builder

処理:
  1. simulation/ptsd_story.txt を読み込みチャンク分割（v1 と同じ入力）
  2. 各チャンクから恐怖構造 fragment を抽出
  3. 刺激/反応/具体的意味/中核評価ノードと結合（EVOKES/MEANS/ROLLS_UP/CO_OCCURS）を登録
  4. エピソードを登録し、刺激→エピソード(RECALLS)・嫌な記憶→嫌な記憶(LEADS_TO)を張る

拡散活性の入口は刺激なので、刺激とエピソードにだけ埋め込みを付ける（反応/意味は構造でたどる）。
"""

import sys

import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import TokenTextSplitter

from .. import config
from ..domain.models import Episode
from ..infrastructure.embeddings import EmbeddingService
from ..infrastructure.neo4j_store import Neo4jStore
from .extractor import FearStructureExtractor


def build(source_path: str = config.SOURCE_TEXT) -> None:
    config.load_env()

    raw = TextLoader(source_path).load()
    chunks = TokenTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    ).split_documents(raw)
    print(f"📄 {source_path} を {len(chunks)} チャンクに分割しました")

    extractor = FearStructureExtractor()
    embedder = EmbeddingService()
    store = Neo4jStore()

    print("🧹 既存の恐怖構造(Fk*)を消去します（v1/simulation グラフは温存）")
    store.clear_fear_graph()
    # 中核評価の層は廃止（意味は開語彙のまま。ensure_core_valuations は呼ばない）。

    order = 0
    n_frag = 0
    stim_emb: dict[str, list] = {}         # 刺激名→埋め込み（SIMILAR 構築に使う）

    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}/{len(chunks)}] 恐怖構造を抽出中...")
        for frag in extractor.extract(chunk.page_content):
            n_frag += 1
            trigger = frag.trigger.strip()
            if not trigger:
                continue

            # 刺激（trigger）＝拡散活性の入口。埋め込みを付ける。
            vec = embedder.embed(trigger).tolist()
            stim_emb[trigger] = vec
            store.merge_node(config.STIMULUS_LABEL, trigger, embedding=vec)

            # 同時に存在した刺激 → CO_OCCURS
            for cue in frag.other_cues:
                cue = cue.strip()
                if not cue or cue == trigger:
                    continue
                cvec = embedder.embed(cue).tolist()
                stim_emb[cue] = cvec
                store.merge_node(config.STIMULUS_LABEL, cue, embedding=cvec)
                store.merge_cooccurs(trigger, cue)

            # 反応 → EVOKES（刺激→反応）
            resp_names = set()
            for resp in frag.responses:
                resp = resp.strip()
                if not resp:
                    continue
                resp_names.add(resp)
                store.merge_node(config.RESPONSE_LABEL, resp)
                store.merge_edge(config.STIMULUS_LABEL, trigger, "EVOKES",
                                 config.RESPONSE_LABEL, resp)

            # 意味づけ → MEANS（刺激 or 反応→意味）。中核評価には丸めない（開語彙）。
            meaning_names = []
            for m in frag.meanings:
                text = m.text.strip()
                if not text:
                    continue
                meaning_names.append(text)
                store.merge_node(config.MEANING_LABEL, text)
                if m.source in resp_names:
                    store.merge_edge(config.RESPONSE_LABEL, m.source, "MEANS",
                                     config.MEANING_LABEL, text)
                else:  # 'trigger' もしくは不明 → 刺激由来として扱う
                    store.merge_edge(config.STIMULUS_LABEL, trigger, "MEANS",
                                     config.MEANING_LABEL, text)

            # エピソード → RECALLS（刺激→記憶）＋ BINDS（記憶↔反応/意味＝S-R-Mの束）
            if frag.episode is not None:
                order += 1
                ep = Episode(
                    id=f"fk_ep_{order:03d}",
                    context=frag.episode.context,
                    event=frag.episode.event,
                    sensory=frag.episode.sensory,
                    valence=(frag.episode.valence or "negative").strip().lower(),
                    t_created=float(order),
                )
                ep.embedding = embedder.embed(ep.as_text()).tolist()
                store.add_episode(ep)
                store.merge_recalls(trigger, ep.id)
                # 記憶＝束ねられた刺激-反応-意味（Lang/Foa）。反応・意味と直接束ねる。
                for rname in resp_names:
                    store.merge_binds(ep.id, config.RESPONSE_LABEL, rname)
                for mname in meaning_names:
                    store.merge_binds(ep.id, config.MEANING_LABEL, mname)

    # 刺激般化：意味的に近い刺激どうしを SIMILAR で結ぶ（汎用則・特定現象向けの手当てはしない）
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
                s = float(sims[i, j])
                if s < config.STIM_SIMILAR_MIN or cnt >= config.STIM_SIMILAR_TOPN:
                    break
                store.merge_similar(a, names[j], s)
                cnt += 1
        print(f"🔗 刺激般化(SIMILAR)を構築しました（min={config.STIM_SIMILAR_MIN}, topN={config.STIM_SIMILAR_TOPN}）")

    counts = store.counts()
    store.close()

    print(f"\n✅ 完了: {n_frag} fragment を処理しました")
    print("--- 恐怖構造グラフ（忠実版オントロジー） ---")
    for label in ["FkStimulus", "FkResponse", "FkMeaning", "FkEpisode"]:
        print(f"  {label:12s}: {counts.get(label, 0)}")
    for rel in ["EVOKES", "MEANS", "CO_OCCURS", "SIMILAR", "RECALLS", "BINDS"]:
        print(f"  {rel:12s}: {counts.get(rel, 0)}")


if __name__ == "__main__":
    # 任意で元テキストのパスを指定可能（未指定なら config.SOURCE_TEXT）。
    build(sys.argv[1] if len(sys.argv) > 1 else config.SOURCE_TEXT)
