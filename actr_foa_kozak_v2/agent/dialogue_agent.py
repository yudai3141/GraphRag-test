"""対話オーケストレーション：1 ターンの処理をまとめる（拡散活性版）。

  発話 → 恐怖構造の拡散活性で連想（SpreadingActivation）
       → DEBUG 出力 → 活性化した部分ネットを文脈に応答生成

v2 Stage 1 は「連想の再現」までなので、記憶（エッジ）の更新はしない（Stage 2 で導入）。
"""

from ..domain.models import FearGraph
from ..infrastructure.llm import LLMClient
from ..prompts import templates
from ..retrieval.spreading import SpreadingActivation
from . import debug_view


class DialogueAgent:
    def __init__(
        self,
        graph: FearGraph,
        spreader: SpreadingActivation,
        llm: LLMClient,
    ):
        self.graph = graph
        self.spreader = spreader
        self.llm = llm

    def respond(self, user_input: str) -> str:
        # 恐怖構造の拡散活性で連想
        result = self.spreader.recall(user_input, self.graph)

        # 内部処理を DEBUG 出力
        debug_view.render(result)

        # 活性化した部分ネットを文脈に応答生成
        user_prompt = templates.build_user_prompt(user_input, result)
        return self.llm.complete(templates.SYSTEM, user_prompt)
