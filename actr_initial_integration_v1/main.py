"""コンソール対話ループ（エントリポイント）。

実行: uv run python -m actr_initial_integration_v1.main

事前に記憶構築が必要:
    uv run python -m actr_initial_integration_v1.memory.builder
"""

from . import config
from .agent.dialogue_agent import DialogueAgent
from .infrastructure.embeddings import EmbeddingService
from .infrastructure.llm import LLMClient
from .infrastructure.neo4j_store import Neo4jStore
from .retrieval.activation import ActrActivationEngine
from .retrieval.recall import MemoryRecaller

_QUIT = {"quit", "exit", "終了", "q"}


def main() -> None:
    config.load_env()

    store = Neo4jStore()
    episodes = store.load_all_episodes()

    if not episodes:
        print("⚠️  エピソード記憶が空です。先に記憶構築を実行してください:")
        print("    uv run python -m actr_initial_integration_v1.memory.builder")
        store.close()
        return

    embedder = EmbeddingService()
    engine = ActrActivationEngine(embedder)
    recaller = MemoryRecaller(engine, store)
    llm = LLMClient()
    agent = DialogueAgent(episodes, store, recaller, llm)

    print("=" * 72)
    print(f"PTSD 対話エージェント (ACT-R 初期統合 v1)")
    print(f"エピソード記憶: {len(episodes)} 件をロードしました")
    print("話しかけてください。終了は quit / exit / 終了。")
    print("=" * 72)

    try:
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
    finally:
        store.close()
        print("\n対話を終了しました。")


if __name__ == "__main__":
    main()
