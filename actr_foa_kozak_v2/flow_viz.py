"""1 クエリの「拡散活性の流れ」を可視化する（pyvis 階層レイアウト）。

  クエリ → 入口の刺激 → 恐怖構造の拡散（刺激/反応/意味）→ 中核評価への収束
        → 想起した過去エピソード → 応答

実データを1クエリ分流して、活性化したノードを到達ホップ順に段レイアウトで並べる。
記憶（グラフ）は変更しない読み取り専用。

  uv run --extra viz python -m actr_foa_kozak_v2.flow_viz "勉強って最近どう？"
"""

import os
import sys

from . import config
from .infrastructure.embeddings import EmbeddingService
from .infrastructure.llm import LLMClient
from .infrastructure.neo4j_store import Neo4jStore
from .prompts import templates
from .retrieval.spreading import SpreadingActivation

_COLOR = {
    config.STIMULUS_LABEL: "#f58231",
    config.RESPONSE_LABEL: "#4363d8",
    config.MEANING_LABEL: "#e6194B",
    config.CORE_LABEL: "#911eb4",
    config.EPISODE_LABEL: "#3cb44b",
}
C_QUERY, C_RESP = "#000000", "#3cb44b"

_LEGEND = (
    '<div style="position:fixed;top:10px;right:10px;z-index:1000;background:rgba(255,255,255,.96);'
    'border:1px solid #ccc;border-radius:8px;padding:10px 12px;font:12px sans-serif;line-height:1.6;">'
    '<b>拡散活性の流れ</b><br>'
    '<span style="color:#000;">■</span> クエリ<br>'
    '<span style="color:#f58231;">●</span> 入口の刺激<br>'
    '<span style="color:#4363d8;">●</span> 反応 / <span style="color:#e6194B;">●</span> 意味<br>'
    '<span style="color:#911eb4;">●</span> 中核評価（収束先）<br>'
    '<span style="color:#3cb44b;">●</span> 想起エピソード → 応答</div>'
)

_OPTIONS = """
{
  "layout": {"hierarchical": {"enabled": true, "direction": "LR",
      "sortMethod": "directed", "levelSeparation": 240, "nodeSpacing": 95}},
  "physics": {"enabled": false},
  "edges": {"smooth": {"type": "cubicBezier", "forceDirection": "horizontal"}}
}
"""


def export_flow(query: str, output_path: str = "flow.html") -> str:
    from pyvis.network import Network

    embedder = EmbeddingService()
    spreader = SpreadingActivation(embedder)
    store = Neo4jStore()
    graph = store.load_fear_graph()
    store.close()
    llm = LLMClient()

    result = spreader.recall(query, graph)
    response = llm.complete(templates.SYSTEM, templates.build_user_prompt(query, result))

    # 表示するノード集合（キー→ ActivatedNode/Episode）と到達ホップ
    shown = {}   # key -> (label, name, activation, hop)
    for s in result.seeds:
        shown[s.node.key] = (s.node.label, s.node.name, s.activation, 0)
    for n in result.nodes:
        shown[n.node.key] = (n.node.label, n.node.name, n.activation, n.hop)
    for a in result.episodes:
        shown[a.episode.key] = (config.EPISODE_LABEL, a.episode.event[:22], a.activation, a.hop)

    max_hop = max((h for _, _, _, h in shown.values()), default=0)

    net = Network(height="840px", width="100%", directed=True, notebook=False)
    net.set_options(_OPTIONS)

    # クエリ（level 0）
    net.add_node("QUERY", label=query, level=0, color=C_QUERY, shape="box",
                 font={"color": "#fff"})
    # 活性ノード（level = hop + 1）
    for key, (label, name, act, hop) in shown.items():
        net.add_node(key, label=f"{name}\n{act:.2f}", title=f"{name} (act={act:.3f}, hop{hop})",
                     level=hop + 1, color=_COLOR.get(label, "#999"),
                     shape="ellipse" if label == config.EPISODE_LABEL else "dot")
    # 入口へ
    for s in result.seeds:
        net.add_edge("QUERY", s.node.key, label=f"入口 {s.activation:.2f}", color="#999")
    # 恐怖構造の実エッジ（表示ノード間のみ）
    for e in graph.edges:
        if e.src in shown and e.dst in shown:
            net.add_edge(e.src, e.dst, label=e.rel, color="#c0c4c8",
                         font={"size": 8, "color": "#888"})
    # 応答（最終段）
    net.add_node("RESPONSE", label=response[:70] + "…", title=response,
                 level=max_hop + 2, color=C_RESP, shape="box")
    for a in result.episodes:
        net.add_edge(a.episode.key, "RESPONSE", color="#3cb44b")
    if not result.episodes:  # エピソードが無ければ中核評価から応答へ
        for n in result.nodes:
            if n.node.label == config.CORE_LABEL:
                net.add_edge(n.node.key, "RESPONSE", color="#3cb44b")

    net.write_html(output_path, notebook=False)
    with open(output_path, "r", encoding="utf-8") as f:
        html = f.read()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html.replace("</body>", _LEGEND + "</body>", 1))

    print(f"Flow written to {output_path}")
    print(f"  クエリ: {query}")
    print(f"  入口: {', '.join(s.node.name for s in result.seeds) or '(なし)'}")
    print(f"  想起エピソード: {', '.join(a.episode.id for a in result.episodes) or '(なし)'}")
    print(f"  応答: {response[:60]}…")
    return output_path


if __name__ == "__main__":
    config.load_env()
    q = sys.argv[1] if len(sys.argv) > 1 else "勉強って最近どう？"
    out = os.path.join(os.path.dirname(__file__), "flow.html")
    export_flow(q, output_path=out)
