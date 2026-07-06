"""恐怖構造グラフを構築して Neo4j に登録する（CLI）。

実行: uv run python -m actr_foa_kozak_v2.memory.builder

処理:
  1. simulation/ptsd_story.txt を読み込みチャンク分割（v1 と同じ入力）
  2. 各チャンクから恐怖構造 fragment を抽出
  3. 刺激/反応/具体的意味/中核評価ノードと結合（EVOKES/MEANS/ROLLS_UP/CO_OCCURS）を登録
  4. エピソードを登録し、刺激→エピソード(RECALLS)・嫌な記憶→嫌な記憶(LEADS_TO)を張る

拡散活性の入口は刺激なので、刺激とエピソードにだけ埋め込みを付ける（反応/意味は構造でたどる）。
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import TokenTextSplitter

from .. import config
from ..domain.models import Episode
from ..infrastructure.embeddings import EmbeddingService
from ..infrastructure.neo4j_store import Neo4jStore
from .extractor import FearStructureExtractor


def build() -> None:
    config.load_env()

    raw = TextLoader(config.SOURCE_TEXT).load()
    chunks = TokenTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    ).split_documents(raw)
    print(f"📄 {config.SOURCE_TEXT} を {len(chunks)} チャンクに分割しました")

    extractor = FearStructureExtractor()
    embedder = EmbeddingService()
    store = Neo4jStore()

    print("🧹 既存の恐怖構造(Fk*)を消去します（v1/simulation グラフは温存）")
    store.clear_fear_graph()
    store.ensure_core_valuations()

    order = 0
    prev_negative_id: str | None = None   # LEADS_TO 用（直前の負エピソード）
    n_frag = 0

    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}/{len(chunks)}] 恐怖構造を抽出中...")
        for frag in extractor.extract(chunk.page_content):
            n_frag += 1
            trigger = frag.trigger.strip()
            if not trigger:
                continue

            # 刺激（trigger）＝拡散活性の入口。埋め込みを付ける。
            store.merge_node(config.STIMULUS_LABEL, trigger,
                             embedding=embedder.embed(trigger).tolist())

            # 同時に存在した刺激 → CO_OCCURS
            for cue in frag.other_cues:
                cue = cue.strip()
                if not cue or cue == trigger:
                    continue
                store.merge_node(config.STIMULUS_LABEL, cue,
                                 embedding=embedder.embed(cue).tolist())
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

            # 意味づけ → MEANS（刺激 or 反応→意味）＋ ROLLS_UP（意味→中核評価）
            for m in frag.meanings:
                text = m.text.strip()
                if not text:
                    continue
                store.merge_node(config.MEANING_LABEL, text)
                store.merge_edge(config.MEANING_LABEL, text, "ROLLS_UP",
                                 config.CORE_LABEL, m.core)
                if m.source in resp_names:
                    store.merge_edge(config.RESPONSE_LABEL, m.source, "MEANS",
                                     config.MEANING_LABEL, text)
                else:  # 'trigger' もしくは不明 → 刺激由来として扱う
                    store.merge_edge(config.STIMULUS_LABEL, trigger, "MEANS",
                                     config.MEANING_LABEL, text)

            # エピソード → RECALLS（刺激→過去の記憶）＋ LEADS_TO（負→負）
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

                # 嫌な記憶は嫌な記憶へ繋がる（良い記憶とは繋がない）
                if ep.valence == "negative":
                    if prev_negative_id is not None:
                        store.merge_leads_to(prev_negative_id, ep.id)
                    prev_negative_id = ep.id

    counts = store.counts()
    store.close()

    print(f"\n✅ 完了: {n_frag} fragment を処理しました")
    print("--- 恐怖構造グラフ ---")
    for label in ["FkStimulus", "FkResponse", "FkMeaning", "FkCore", "FkEpisode"]:
        print(f"  {label:12s}: {counts.get(label, 0)}")
    for rel in ["EVOKES", "MEANS", "ROLLS_UP", "RECALLS", "LEADS_TO", "CO_OCCURS"]:
        print(f"  {rel:12s}: {counts.get(rel, 0)}")


if __name__ == "__main__":
    build()
