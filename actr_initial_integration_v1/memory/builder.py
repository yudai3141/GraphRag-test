"""エピソード記憶グラフを構築して Neo4j に登録する（CLI）。

実行: uv run python -m actr_initial_integration_v1.memory.builder

処理:
  1. simulation/ptsd_story.txt を読み込みチャンク分割
  2. 各チャンクから構造化エピソードを抽出
  3. 各エピソードに B(m)=ランダム・t_created=物語順の論理時刻・埋め込みを付与
  4. Neo4j に Episode / MemEntity として登録し、NEXT で時系列連結
"""

import random

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import TokenTextSplitter

from .. import config
from ..domain.models import Episode, MemEntity
from ..infrastructure.embeddings import EmbeddingService
from ..infrastructure.neo4j_store import Neo4jStore
from .extractor import EpisodeExtractor


def build() -> None:
    config.load_env()

    raw = TextLoader(config.SOURCE_TEXT).load()
    chunks = TokenTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    ).split_documents(raw)
    print(f"📄 {config.SOURCE_TEXT} を {len(chunks)} チャンクに分割しました")

    extractor = EpisodeExtractor()
    embedder = EmbeddingService()
    store = Neo4jStore()

    print("🧹 既存のエピソード記憶(Episode/MemEntity)を消去します（simulationグラフは温存）")
    store.clear_episodic_memory()

    episodes: list[Episode] = []
    order = 0
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}/{len(chunks)}] エピソード抽出中...")
        for ex in extractor.extract(chunk.page_content):
            order += 1
            ep = Episode(
                id=f"ep_{order:03d}",
                context=ex.context,
                event=ex.event,
                sensory=ex.sensory,
                b_m=random.gauss(config.BM_INIT_MEAN, config.BM_INIT_STD),
                t_created=float(order),          # 物語順を論理時刻として使う（将来は実時刻に置換）
                presentations=[float(order)],    # 記銘＝最初の提示
                entities=[MemEntity(**e.model_dump()) for e in ex.entities],
            )
            ep.embedding = embedder.embed(ep.as_text()).tolist()
            store.add_episode(ep)
            episodes.append(ep)

    # 時系列連結
    for prev, nxt in zip(episodes, episodes[1:]):
        store.link_next(prev.id, nxt.id)

    # 意味層（エンティティ共起ネットワーク）を構築
    assoc_count = store.build_entity_associations()
    print(f"🔗 意味層(ASSOC)を構築: {assoc_count} 本のエンティティ連想")

    store.close()
    print(f"\n✅ 完了: {len(episodes)} エピソードを登録しました")
    if episodes:
        print("--- 例 ---")
        for ep in episodes[:3]:
            ents = ", ".join(f"{e.name}({e.type})" for e in ep.entities)
            print(f"  {ep.id} | B(m)初期={ep.b_m:+.2f}")
            print(f"     {ep.as_text()}")
            print(f"     entities: {ents}")


if __name__ == "__main__":
    build()
