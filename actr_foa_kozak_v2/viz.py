"""恐怖構造グラフを pyvis でインタラクティブ HTML に書き出す。

  uv run --extra viz python -m actr_foa_kozak_v2.viz

刺激=橙 / 反応=青 / 具体的意味=赤 / 中核評価=紫 / エピソード=緑 で色分けする。
具体的意味づけが少数の中核評価へ収束していく様子（PTSDの構造）が見える。
"""

import os

from neo4j import GraphDatabase

from . import config

# 恐怖構造全体（Fk* 間の関係）を描く。密なら LIMIT を下げる。
DEFAULT_CYPHER = """
MATCH (a)-[r]->(b)
WHERE any(l IN labels(a) WHERE l STARTS WITH 'Fk')
  AND any(l IN labels(b) WHERE l STARTS WITH 'Fk')
RETURN a, r, b
LIMIT 500
"""

_COLOR = {
    config.STIMULUS_LABEL: "#f58231",   # 橙
    config.RESPONSE_LABEL: "#4363d8",   # 青
    config.MEANING_LABEL: "#e6194B",    # 赤
    config.CORE_LABEL: "#911eb4",       # 紫
    config.EPISODE_LABEL: "#3cb44b",    # 緑
}

_LEGEND = (
    '<div style="position:fixed;top:10px;right:10px;z-index:1000;background:rgba(255,255,255,.95);'
    'border:1px solid #ccc;border-radius:8px;padding:8px 12px;font:12px sans-serif;line-height:1.6;">'
    '<b>恐怖構造（Foa & Kozak）</b><br>'
    '<span style="color:#f58231;">●</span> 刺激 Stimulus<br>'
    '<span style="color:#4363d8;">●</span> 反応 Response<br>'
    '<span style="color:#e6194B;">●</span> 具体的意味 Meaning<br>'
    '<span style="color:#911eb4;">●</span> 中核評価 Core<br>'
    '<span style="color:#3cb44b;">●</span> エピソード Episode</div>'
)


def _label_of(node) -> str:
    for l in node.labels:
        if l.startswith("Fk"):
            return l
    return next(iter(node.labels), "")


def export_html(cypher: str = DEFAULT_CYPHER, output_path: str = "graph.html") -> str:
    from pyvis.network import Network

    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    net = Network(height="820px", width="100%", directed=True, notebook=False)

    with driver.session(database=os.environ.get("NEO4J_DATABASE", "neo4j")) as session:
        graph = session.run(cypher).graph()
        for node in graph.nodes:
            label = _label_of(node)
            if label == config.EPISODE_LABEL:
                text = (node.get("event") or node.get("name") or "")[:18]
                title = (f"{node.get('name')}[{node.get('valence')}]\n"
                         f"状況: {node.get('context')}\n出来事: {node.get('event')}")
            elif label == config.CORE_LABEL:
                text = node.get("jp") or node.get("name") or ""
                title = f"中核評価: {text}"
            else:
                text = node.get("name") or ""
                title = f"{text} ({label})"
            net.add_node(node.element_id, label=str(text), title=title,
                         color=_COLOR.get(label, "#999"), shape="dot")
        for rel in graph.relationships:
            net.add_edge(rel.start_node.element_id, rel.end_node.element_id,
                         label=rel.type, color="#9aa0a6", font={"size": 9, "color": "#555"})
    driver.close()

    net.write_html(output_path, notebook=False)
    with open(output_path, "r", encoding="utf-8") as f:
        html = f.read()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html.replace("</body>", _LEGEND + "</body>", 1))
    print(f"Graph written to {output_path}")
    return output_path


if __name__ == "__main__":
    config.load_env()
    out = os.path.join(os.path.dirname(__file__), "graph.html")
    export_html(output_path=out)
