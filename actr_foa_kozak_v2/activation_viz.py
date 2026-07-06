"""クエリを投げると、恐怖構造グラフ全体の上に「どこが活性化したか」を重ねて表示する。

  uv run --extra viz python -m actr_foa_kozak_v2.activation_viz "電車ってよく乗るの？"

全ノードを力学レイアウトで描き、拡散活性で活性化したノードだけを役割色＋活性に応じた
大きさ・明るさで強調する（非活性は薄いグレーの小ノード）。クエリを変えると光る場所が動く。
viz.py（静的な全体図）と違い、こちらは「クエリに対する活性ヒートマップ」を重ねたもの。
"""

import os
import sys

from . import config
from .domain.models import node_key
from .infrastructure.embeddings import EmbeddingService
from .infrastructure.neo4j_store import Neo4jStore
from .retrieval.spreading import SpreadingActivation

# 役割ごとの色（viz.py と統一）
_COLOR = {
    config.STIMULUS_LABEL: "#f58231",   # 橙
    config.RESPONSE_LABEL: "#4363d8",   # 青
    config.MEANING_LABEL: "#e6194B",    # 赤
    config.CORE_LABEL: "#911eb4",       # 紫
    config.EPISODE_LABEL: "#3cb44b",    # 緑
}
_DIM = "#e3e3e3"                        # 非活性ノードの色

_OPTIONS = """
{
  "physics": {"solver": "forceAtlas2Based",
    "forceAtlas2Based": {"gravitationalConstant": -45, "springLength": 110, "avoidOverlap": 0.6},
    "stabilization": {"iterations": 250}},
  "edges": {"smooth": false}
}
"""


def _legend(query: str, core_line: str) -> str:
    return (
        '<div style="position:fixed;top:10px;right:10px;z-index:1000;background:rgba(255,255,255,.96);'
        'border:1px solid #ccc;border-radius:8px;padding:10px 12px;font:12px sans-serif;line-height:1.6;">'
        f'<b>活性ヒートマップ</b><br><span style="color:#555;">query:</span> {query}<br>'
        f'<span style="color:#555;">収束:</span> {core_line or "-"}<hr style="margin:6px 0;border:none;border-top:1px solid #eee;">'
        '<span style="color:#f58231;">●</span> 刺激 Stimulus<br>'
        '<span style="color:#4363d8;">●</span> 反応 Response<br>'
        '<span style="color:#e6194B;">●</span> 具体的意味 Meaning<br>'
        '<span style="color:#911eb4;">●</span> 中核評価 Core<br>'
        '<span style="color:#3cb44b;">●</span> エピソード Episode<br>'
        '<span style="color:#e3e3e3;">●</span> 非活性（薄いグレー）<hr style="margin:6px 0;border:none;border-top:1px solid #eee;">'
        '大きさ・濃さ＝活性の強さ / 太い枠＝入口の刺激</div>'
    )


def core_summary(graph, activation) -> str:
    """中核評価の収束サマリ（凡例・字幕用）。"""
    return " / ".join(
        f"{n.name}={activation[n.key]:.2f}"
        for n in sorted(
            [n for n in graph.nodes if n.label == config.CORE_LABEL and n.key in activation],
            key=lambda n: activation[n.key], reverse=True,
        )
    )


def render_html(graph, activation, reached_hop, seeds, query: str, height: str = "860px") -> str:
    """恐怖構造グラフ全体に活性を重ねた HTML 文字列を返す（CLI・UI で共用）。"""
    from pyvis.network import Network

    seed_keys = {s.key for s, _ in seeds}
    max_act = max(activation.values(), default=1.0) or 1.0

    net = Network(height=height, width="100%", directed=True, notebook=False)
    net.set_options(_OPTIONS)

    all_nodes = [(n.key, n.label, n.name) for n in graph.nodes] + \
                [(e.key, config.EPISODE_LABEL, e.event[:20]) for e in graph.episodes]
    for key, label, name in all_nodes:
        act = activation.get(key, 0.0)
        if act > 1e-4:
            size = 12 + 40 * (act / max_act)
            color = _COLOR.get(label, "#999")
            border = 4 if key in seed_keys else 1
            net.add_node(key, label=name, title=f"{name}\n活性={act:.3f} (hop{reached_hop.get(key,'-')})",
                         color={"background": color, "border": "#222"},
                         size=size, borderWidth=border, font={"size": 14})
        else:
            net.add_node(key, label=" ", title=name, color=_DIM, size=5,
                         borderWidth=0, font={"size": 1})

    for e in graph.edges:
        both = activation.get(e.src, 0) > 1e-4 and activation.get(e.dst, 0) > 1e-4
        if both:
            net.add_edge(e.src, e.dst, label=e.rel, color="#555",
                         width=2, font={"size": 9, "color": "#555"})
        else:
            net.add_edge(e.src, e.dst, color="#efefef", width=0.3)

    html = net.generate_html(notebook=False)
    return html.replace("</body>", _legend(query, core_summary(graph, activation)) + "</body>", 1)


def export_activation(query: str, output_path: str = "activation.html") -> str:
    embedder = EmbeddingService()
    spreader = SpreadingActivation(embedder)
    store = Neo4jStore()
    graph = store.load_fear_graph()
    store.close()

    activation, reached_hop, seeds = spreader.compute(query, graph)
    html = render_html(graph, activation, reached_hop, seeds, query)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    n_active = sum(1 for v in activation.values() if v > 1e-4)
    print(f"Activation map written to {output_path}")
    print(f"  クエリ: {query}")
    print(f"  入口: {', '.join(s.name for s, _ in seeds) or '(なし)'}")
    print(f"  活性ノード数: {n_active}")
    print(f"  中核収束: {core_summary(graph, activation) or '-'}")
    return output_path


if __name__ == "__main__":
    config.load_env()
    q = sys.argv[1] if len(sys.argv) > 1 else "電車ってよく乗るの？"
    out = os.path.join(os.path.dirname(__file__), "activation.html")
    export_activation(q, output_path=out)
