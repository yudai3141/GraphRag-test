"""対話オーケストレーション：1 ターンの処理をまとめる（二段想起版）。

  発話 → ① ACT-R で seed 特定 → ② エンティティ連想で関連エピソード取得（MemoryRecaller）
       → DEBUG 出力 → seed＋関連を文脈に応答生成 → 想起エピソードを強化
"""

from typing import List

from .. import config
from ..domain.models import Episode
from ..infrastructure.llm import LLMClient
from ..infrastructure.neo4j_store import Neo4jStore
from ..prompts import templates
from ..retrieval.recall import MemoryRecaller
from . import debug_view


class DialogueAgent:
    def __init__(
        self,
        episodes: List[Episode],
        store: Neo4jStore,
        recaller: MemoryRecaller,
        llm: LLMClient,
        threshold: float = config.RETRIEVAL_THRESHOLD,
    ):
        self.episodes = episodes
        self.store = store
        self.recaller = recaller
        self.llm = llm
        self.threshold = threshold
        # 論理時刻：記銘済みの最大時刻の次から開始し、ターンごとに進める
        self.clock = max((ep.t_created for ep in episodes), default=0.0) + 1.0

    def respond(self, user_input: str) -> str:
        self.clock += 1.0
        now = self.clock

        # ①②: seed 特定 ＋ 連想
        result = self.recaller.recall(user_input, self.episodes, now)

        # 内部処理を DEBUG 出力
        debug_view.render(result, self.threshold, now)

        # 文脈：seed ＋ 関連エピソード
        seed_ep = result.seed.episode if result.seed else None
        related_eps = [a.episode for a in result.related]

        user_prompt = templates.build_user_prompt(user_input, seed_ep, related_eps)
        response = self.llm.complete(templates.SYSTEM, user_prompt)

        # 想起したエピソードを強化（次回以降 B(m) が上がる）
        recalled = ([seed_ep] if seed_ep else []) + related_eps
        for ep in recalled:
            ep.presentations.append(now)
            self.store.reinforce(ep.id, now)

        return response
