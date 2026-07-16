"""スナップショット(JSON)から恐怖記憶グラフの静止画 PNG を描く（報告書埋め込み用）。

実行: uv run --with matplotlib --with networkx python -m caps5_graph_structure_eval_v3.graph_image \
        caps5_graph_structure_eval_v3/data/graph_exp_ptsd.json caps5_graph_structure_eval_v3/figures/graph_ptsd_view.png "PTSD graph"
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from actr_foa_kozak_v2 import config

COLORS = {
    config.STIMULUS_LABEL: "#f58231",   # 橙 刺激
    config.RESPONSE_LABEL: "#4363d8",   # 青 反応
    config.MEANING_LABEL: "#e6194B",    # 赤 意味
    config.EPISODE_LABEL: "#2E8B3D",    # 緑 エピソード
}
LEGEND = [("#f58231", "stimulus"), ("#4363d8", "response"),
          ("#e6194B", "meaning"), ("#2E8B3D", "episode (memory)")]


def render(path, out, title):
    g = json.load(open(path))
    G = nx.Graph()
    ncolor, nsize = {}, {}
    for n in g["nodes"]:
        G.add_node(n["key"]); ncolor[n["key"]] = COLORS.get(n["label"], "#999"); nsize[n["key"]] = 22
    for e in g["episodes"]:
        k = f"FkEpisode::{e['id']}"
        G.add_node(k); ncolor[k] = COLORS[config.EPISODE_LABEL]; nsize[k] = 70
    for ed in g["edges"]:
        if ed["src"] in ncolor and ed["dst"] in ncolor:
            G.add_edge(ed["src"], ed["dst"])
    print(f"  layout: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges ...")
    pos = nx.spring_layout(G, k=0.32, iterations=70, seed=42)

    fig, ax = plt.subplots(figsize=(9, 9))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#cdd2d6", width=0.35, alpha=0.45)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=list(G.nodes()),
                           node_color=[ncolor[n] for n in G.nodes()],
                           node_size=[nsize[n] for n in G.nodes()], alpha=0.92, linewidths=0)
    ax.set_title(title, fontsize=15, color="#26323B", fontweight="bold")
    ax.axis("off")
    ax.legend(handles=[mpatches.Patch(color=c, label=l) for c, l in LEGEND],
              loc="lower left", frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ {out}")


if __name__ == "__main__":
    config.load_env()
    render(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
