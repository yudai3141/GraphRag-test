"""スナップショット(JSON)から恐怖構造グラフを pyvis で HTML 描画する（Neo4j 不要）。

論文・スライド用のグラフ写真を撮るためのもの。忠実版オントロジー（中核層なし）の
色分け：刺激=橙 / 反応=青 / 意味=赤 / エピソード=緑。

実行例:
  uv run --extra viz python -m caps5_graph_structure_eval_v3.viz_snapshot                      # graph_faithful.json
  uv run --extra viz python -m caps5_graph_structure_eval_v3.viz_snapshot data/graph_plus_daily.json
  uv run --extra viz python -m caps5_graph_structure_eval_v3.viz_snapshot data/graph_faithful.json out.html
"""

import os
import sys

from actr_foa_kozak_v2 import config

from . import snapshot

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_COLOR = {
    config.STIMULUS_LABEL: "#f58231",   # 橙
    config.RESPONSE_LABEL: "#4363d8",   # 青
    config.MEANING_LABEL: "#e6194B",    # 赤
    config.EPISODE_LABEL: "#3cb44b",    # 緑
}

_LEGEND = (
    '<div style="position:fixed;top:12px;right:12px;z-index:1000;background:rgba(255,255,255,.96);'
    'border:1px solid #ccc;border-radius:8px;padding:10px 14px;font:13px sans-serif;line-height:1.7;">'
    '<b>記憶グラフ（Foa & Kozak 恐怖構造 × エピソード記憶）</b><br>'
    '<span style="color:#f58231;">●</span> 刺激 Stimulus<br>'
    '<span style="color:#4363d8;">●</span> 反応 Response<br>'
    '<span style="color:#e6194B;">●</span> 意味 Meaning<br>'
    '<span style="color:#3cb44b;">●</span> エピソード Episode</div>'
)


def export(graph_path: str, output_path: str, show_labels: bool = True) -> str:
    from pyvis.network import Network

    graph = snapshot.load(graph_path)
    net = Network(height="900px", width="100%", directed=False, notebook=False, bgcolor="#ffffff")
    net.barnes_hut(gravity=-12000, spring_length=110, spring_strength=0.02)

    ep_ids = {e.key for e in graph.episodes}
    ep_by_key = {e.key: e for e in graph.episodes}
    for n in graph.nodes:
        text = n.name if show_labels else " "
        net.add_node(n.key, label=str(text)[:20], title=f"{n.name} ({n.label})",
                     color=_COLOR.get(n.label, "#999"), shape="dot", size=12,
                     font={"size": 11, "color": "#333"})
    for k in ep_ids:
        e = ep_by_key[k]
        text = (e.event or e.id)[:20] if show_labels else " "
        net.add_node(k, label=str(text), title=f"[{e.valence}] {e.event}",
                     color=_COLOR[config.EPISODE_LABEL], shape="dot", size=16,
                     font={"size": 11, "color": "#333"})
    known = {n.key for n in graph.nodes} | ep_ids
    for ed in graph.edges:
        if ed.src in known and ed.dst in known:
            net.add_edge(ed.src, ed.dst, label=ed.rel,
                         color="#c8ccd0", font={"size": 8, "color": "#888"})

    net.write_html(output_path, notebook=False)
    with open(output_path, encoding="utf-8") as f:
        html = f.read()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html.replace("</body>", _LEGEND + "</body>", 1))
    print(f"✅ {len(graph.nodes)+len(graph.episodes)} ノード / {len(graph.edges)} エッジ → {output_path}")
    return output_path


def main() -> None:
    config.load_env()
    gp = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "data", "graph_faithful.json")
    if not os.path.isabs(gp):
        gp = os.path.join(_HERE, gp) if not gp.startswith("caps5_graph_structure_eval_v3") else gp
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_HERE, "graph_view.html")
    export(gp, out)


if __name__ == "__main__":
    main()
