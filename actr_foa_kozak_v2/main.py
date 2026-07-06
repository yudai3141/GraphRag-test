"""コンソール対話ループ（エントリポイント・v2）。

実行: uv run python -m actr_foa_kozak_v2.main

事前に恐怖構造グラフの構築が必要:
    uv run python -m actr_foa_kozak_v2.memory.builder
"""

from . import config
from .agent.dialogue_agent import DialogueAgent
from .infrastructure.embeddings import EmbeddingService
from .infrastructure.llm import LLMClient
from .infrastructure.neo4j_store import Neo4jStore
from .retrieval.spreading import SpreadingActivation

_QUIT = {"quit", "exit", "終了", "q"}


def main() -> None:
    config.load_env()

    store = Neo4jStore()
    graph = store.load_fear_graph()
    store.close()   # 以降はメモリ上のグラフで拡散活性する（読み取り専用）

    if not graph.stimuli():
        print("⚠️  恐怖構造グラフが空です。先に構築を実行してください:")
        print("    uv run python -m actr_foa_kozak_v2.memory.builder")
        return

    embedder = EmbeddingService()
    spreader = SpreadingActivation(embedder)
    llm = LLMClient()
    agent = DialogueAgent(graph, spreader, llm)

    print("=" * 72)
    print("PTSD 対話エージェント (Foa & Kozak 恐怖構造 v2)")
    print(f"恐怖構造: 刺激{len(graph.stimuli())} / 全ノード{len(graph.nodes)} / "
          f"エピソード{len(graph.episodes)} / エッジ{len(graph.edges)}")
    print("話しかけてください。終了は quit / exit / 終了。")
    print("=" * 72)

    while True:
        try:
            user_input = input("\nあなた > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() in _QUIT:
            break

        response = agent.respond(user_input)
        print(f"\n本人 > {response}")

    print("\n対話を終了しました。")


if __name__ == "__main__":
    main()
