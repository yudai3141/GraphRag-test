"""恐怖構造グラフのスナップショット（Neo4j → JSON 保存 / 読込）。

実験は Neo4j に依存せず、保存した JSON から FearGraph を復元して回す。
これにより PTSD 版・健常版など複数の「記憶の地図」を並存させて比較できる
（Neo4j 側は builder が Fk* を消して作り直すため、同時に1つしか持てない）。

実行: uv run python -m evaluation_v3.snapshot [出力パス]
（既定: evaluation_v3/data/graph_ptsd.json）
"""

import json
import os
import sys

from actr_foa_kozak_v2 import config
from actr_foa_kozak_v2.domain.models import Edge, Episode, FearGraph, FearNode
from actr_foa_kozak_v2.infrastructure.neo4j_store import Neo4jStore

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(_HERE, "data", "graph_ptsd.json")


def dump(graph: FearGraph, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {
        "nodes": [n.model_dump() for n in graph.nodes],
        "episodes": [e.model_dump() for e in graph.episodes],
        "edges": [e.model_dump() for e in graph.edges],
    }
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False)


def load(path: str) -> FearGraph:
    with open(path) as f:
        data = json.load(f)
    return FearGraph(
        nodes=[FearNode(**n) for n in data["nodes"]],
        episodes=[Episode(**e) for e in data["episodes"]],
        edges=[Edge(**e) for e in data["edges"]],
    )


def main() -> None:
    config.load_env()
    out = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    store = Neo4jStore()
    graph = store.load_fear_graph()
    store.close()
    dump(graph, out)
    print(f"✅ ノード {len(graph.nodes)} / エピソード {len(graph.episodes)} / "
          f"エッジ {len(graph.edges)} → {out}")


if __name__ == "__main__":
    main()