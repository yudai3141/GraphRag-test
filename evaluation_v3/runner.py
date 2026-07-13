"""一括測定ランナー：ひとことリスト × 比較条件 で拡散活性を回し、指標を CSV に記録する。

忠実版オントロジー（中核層なし・無向対称・重み一律）での測定。
怖がり度は「反応要素(FkResponse)の活性」から読む（生体情報理論：感情の中心は反応）。
グラフはスナップショット(JSON)から読むので Neo4j は不要。OpenAI はキューの埋め込みのみ。

比較条件:
  faithful   … 忠実版そのまま（無向対称・重み1.0・危険引き寄せなし。config 既定）
  one_way    … 逆方向を止める(backward=0)。双方向＝侵入想起の寄与を見るアブレーション
  no_similar … 刺激般化(SIMILAR)を除去
  no_bind    … 記憶の束ね(BINDS)を除去。エピソード↔S-R-M の寄与を見る
  no_spread  … 拡散なし・入口の類似度のみ（≒ベクタ検索/GraphRAG 相当）

実行: uv run python -m evaluation_v3.runner [グラフJSON] [バッテリーJSON]
出力: evaluation_v3/results/metrics.csv（1キュー×1条件=1行）
"""

import csv
import os
import sys
from typing import Dict, Tuple

from actr_foa_kozak_v2 import config
from actr_foa_kozak_v2.domain.models import FearGraph
from actr_foa_kozak_v2.infrastructure.embeddings import EmbeddingService
from actr_foa_kozak_v2.retrieval.spreading import SpreadingActivation

from . import cue_battery, snapshot

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(_HERE, "results", "metrics.csv")

FIELDS = [
    "cue", "level", "condition",
    "seed_top_sim", "n_seeds",
    "resp_act",       # 怖がり度＝反応要素の活性（新・主指標）
    "meaning_act",    # 恐怖の「内容」＝意味要素の活性（補助）
    "neg_ep_act",     # 侵入記憶＝負エピソードの活性
    "pos_ep_act",     # 回復の入口＝正エピソードの活性
    "triggered",
]


def _strip(graph: FearGraph, rel: str) -> FearGraph:
    return FearGraph(
        nodes=graph.nodes, episodes=graph.episodes,
        edges=[e for e in graph.edges if e.rel != rel],
    )


def build_conditions(embedder: EmbeddingService, graph: FearGraph
                     ) -> Dict[str, Tuple[SpreadingActivation, FearGraph]]:
    # faithful は config 既定（backward=1.0, danger_bias=0.0, 重み一律）を使う。
    return {
        "faithful": (SpreadingActivation(embedder), graph),
        "one_way": (SpreadingActivation(embedder, backward=0.0), graph),
        "no_similar": (SpreadingActivation(embedder), _strip(graph, "SIMILAR")),
        "no_bind": (SpreadingActivation(embedder), _strip(graph, "BINDS")),
        "no_spread": (SpreadingActivation(embedder, hops=0), graph),
    }


def measure(engine: SpreadingActivation, graph: FearGraph, cue: str) -> dict:
    """1 キューの拡散活性を回し、指標を集計して返す。"""
    by_key = {n.key: n for n in graph.nodes}
    ep_by_key = {e.key: e for e in graph.episodes}

    activation, _reached, seeds = engine.compute(cue, graph)

    resp = meaning = 0.0
    eps = []  # (activation, valence)
    for key, a in activation.items():
        node = by_key.get(key)
        if node is not None:
            if node.label == config.RESPONSE_LABEL:
                resp += a
            elif node.label == config.MEANING_LABEL:
                meaning += a
        else:
            ep = ep_by_key.get(key)
            if ep is not None:
                eps.append((a, ep.valence))

    neg = sum(a for a, v in eps if v == "negative")
    pos = sum(a for a, v in eps if v == "positive")

    # 引き金ゲート（入口が十分一致 かつ 上位想起が負優位）。中核層に依存しない。
    seed_top = seeds[0][1] if seeds else 0.0
    top_eps = sorted(eps, key=lambda x: x[0], reverse=True)[: config.TOP_EPISODES]
    top_neg = sum(a for a, v in top_eps if v == "negative")
    top_pos = sum(a for a, v in top_eps if v == "positive")
    triggered = seed_top >= config.TRIGGER_SIM_MIN and top_neg > top_pos

    return {
        "seed_top_sim": round(seed_top, 4),
        "n_seeds": len(seeds),
        "resp_act": round(resp, 4),
        "meaning_act": round(meaning, 4),
        "neg_ep_act": round(neg, 4),
        "pos_ep_act": round(pos, 4),
        "triggered": int(triggered),
    }


def main() -> None:
    config.load_env()
    graph_path = sys.argv[1] if len(sys.argv) > 1 else snapshot.DEFAULT_PATH
    battery_path = sys.argv[2] if len(sys.argv) > 2 else cue_battery.DEFAULT_PATH

    graph = snapshot.load(graph_path)
    cues = cue_battery.load(battery_path)
    embedder = EmbeddingService()
    conditions = build_conditions(embedder, graph)
    print(f"📊 グラフ: {os.path.basename(graph_path)} / キュー {len(cues)} 個 × 条件 {len(conditions)}")

    rows = []
    for i, c in enumerate(cues, 1):
        for cond, (engine, g) in conditions.items():
            m = measure(engine, g, c["text"])
            rows.append({"cue": c["text"], "level": c["level"], "condition": cond, **m})
        if i % 20 == 0:
            print(f"  {i}/{len(cues)} キュー完了")

    os.makedirs(os.path.dirname(DEFAULT_OUT), exist_ok=True)
    with open(DEFAULT_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ {len(rows)} 行を書き出し → {DEFAULT_OUT}")

    conds = list(conditions)
    levels = sorted({r["level"] for r in rows})

    def table(metric: str, title: str) -> None:
        print(f"\n--- レベル × 条件: {title} ---")
        print("level | " + " | ".join(f"{c:>12s}" for c in conds))
        for lv in levels:
            cells = []
            for cond in conds:
                sub = [r for r in rows if r["level"] == lv and r["condition"] == cond]
                mean_v = sum(r[metric] for r in sub) / max(len(sub), 1)
                cells.append(f"{mean_v:12.3f}")
            print(f"  L{lv}  | " + " | ".join(cells))

    # 怖がり度＝反応活性（主指標）と引き金率、侵入記憶＝負エピソード活性
    table("resp_act", "怖がり度＝反応活性（主指標）")
    print("\n--- レベル × 条件: 引き金率 ---")
    print("level | " + " | ".join(f"{c:>12s}" for c in conds))
    for lv in levels:
        cells = []
        for cond in conds:
            sub = [r for r in rows if r["level"] == lv and r["condition"] == cond]
            rate = sum(r["triggered"] for r in sub) / max(len(sub), 1)
            cells.append(f"{rate:11.0%} ")
        print(f"  L{lv}  | " + " | ".join(cells))
    table("neg_ep_act", "侵入記憶＝負エピソード活性")


if __name__ == "__main__":
    main()
