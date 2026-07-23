"""発話を1つ入れて、恐怖記憶グラフから「何が取り出され、どんな発話が出るか」を1本のトレースにする。

Neo4j 非依存（スナップショット JSON からグラフを読む）。応答生成の実装は
`actr_foa_kozak_v2`（拡散活性・引き金ゲート・プロンプト）をそのまま流用する。

  trace = probe(query, graph, spreader, llm)

戻り値 trace（dict）：
  query      … 入力発話
  seeds      … 入口になった刺激 [(名前, 類似度), ...]
  responses  … 活性化した反応（からだ） [(名前, 活性), ...]
  meanings   … 活性化した意味 [(名前, 活性), ...]
  episodes   … 想起した過去 [(出来事, valence, 活性), ...]
  triggered  … 引き金が立ったか（True=破局モード / False=平静モード）
  response   … 生成された発話（llm=None なら None）

可視化(render_demo)・ライブUI(app)・例文生成が、この同じトレースを使う。
"""

from typing import List, Optional

from actr_foa_kozak_v2 import config
from actr_foa_kozak_v2.domain.models import FearGraph, RecallResult
from actr_foa_kozak_v2.prompts import templates
from actr_foa_kozak_v2.retrieval.spreading import SpreadingActivation


def _top(result: RecallResult, label: str, k: int):
    return [(n.node.name, round(n.activation, 3))
            for n in result.nodes if n.node.label == label][:k]


def probe(query: str, graph: FearGraph, spreader: SpreadingActivation,
          llm: Optional[object] = None) -> dict:
    """1 発話ぶんの拡散活性を回し、抽出結果と（任意で）生成応答をまとめて返す。"""
    result: RecallResult = spreader.recall(query, graph)
    triggered = templates.is_triggered(result)

    trace = {
        "query": query,
        "seeds": [(s.node.name, round(s.activation, 3)) for s in result.seeds],
        "responses": _top(result, config.RESPONSE_LABEL, 4),
        "meanings": _top(result, config.MEANING_LABEL, 4),
        "episodes": [(a.episode.event, a.episode.valence, round(a.activation, 3))
                     for a in result.episodes[:4]],
        "triggered": triggered,
        "response": None,
    }

    if llm is not None:
        user_prompt = templates.build_user_prompt(query, result)
        trace["response"] = llm.complete(templates.SYSTEM, user_prompt)

    return trace


# 初回面接で臨床家が投げそうな発話（レベルを散らして平静／破局の両方が出るようにしてある）
DEFAULT_BATTERY: List[str] = [
    "今日はお話しする時間をとってくれてありがとう。今、どんなことで困っていますか？",
    "夜は眠れていますか？",
    "眠れないんですね。原因に心当たりはありますか？",
    "電車には乗れていますか？",
    "人混みはどうですか？",
    "最近、気分はどんな感じですか？",
    "これから、どんなふうになっていきたいですか？",
    "今日は天気がいいですね。ここまで歩いてこられましたか？",
]
