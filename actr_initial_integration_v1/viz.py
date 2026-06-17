"""エピソード記憶グラフを pyvis でインタラクティブ HTML に書き出す。

  uv run --extra viz python -m actr_initial_integration_v1.viz

Episode は青・MemEntity は赤で色分けし、Episode はホバーで context/event/sensory を表示する。
既定では電車クラスタ周辺の小さな部分グラフを描く（全体は密すぎて見づらいため）。
"""

import os

from neo4j import GraphDatabase

from . import config

# 既定：読みやすい部分グラフ（電車関連エピソード＋そのエンティティ＋ASSOC）
DEFAULT_CYPHER = """
MATCH (e:Episode)-[r1]->(m:MemEntity)
WHERE e.id IN ['ep_003','ep_004','ep_005','ep_056']
OPTIONAL MATCH (m)-[a:ASSOC]-(m2:MemEntity)
RETURN e, r1, m, a, m2
LIMIT 300
"""

_LEGEND = (
    '<div style="position:fixed;top:10px;right:10px;z-index:1000;background:rgba(255,255,255,.95);'
    'border:1px solid #ccc;border-radius:8px;padding:8px 12px;font:12px sans-serif;">'
    '<b>凡例</b><br>'
    '<span style="color:#4363d8;">●</span> Episode（エピソード記憶）<br>'
    '<span style="color:#e6194B;">●</span> MemEntity（エンティティ）<br>'
    '<span style="color:#777;">─</span> エッジ＝関係/ASSOC</div>'
)


def export_html(cypher: str = DEFAULT_CYPHER, output_path: str = "graph.html") -> str:
    from pyvis.network import Network

    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    net = Network(height="800px", width="100%", directed=True, notebook=False)

    with driver.session(database=os.environ.get("NEO4J_DATABASE", "neo4j")) as session:
        graph = session.run(cypher).graph()
        for node in graph.nodes:
            labels = list(node.labels)
            if config.EPISODE_LABEL in labels:
                label = (node.get("event") or node.get("id") or "")[:18]
                title = (
                    f"{node.get('id')}\n状況: {node.get('context')}\n"
                    f"出来事: {node.get('event')}\n感覚: {node.get('sensory')}"
                )
                color, shape = "#4363d8", "dot"
            else:
                label = node.get("name") or ""
                title = f"{node.get('name')} ({node.get('type')})"
                color, shape = "#e6194B", "dot"
            net.add_node(
                node.element_id, label=str(label), title=title, color=color, shape=shape
            )
        for rel in graph.relationships:
            net.add_edge(
                rel.start_node.element_id, rel.end_node.element_id,
                label=rel.type, color="#9aa0a6", font={"size": 9, "color": "#555"},
            )
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
