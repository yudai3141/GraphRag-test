"""全体グラフ と 一発話で一部だけ活性化したグラフ を「同じレイアウト」で左右に並べた PNG を描く。

  uv run --with matplotlib --with networkx python -m caps5_graph_structure_eval_v3.demo.render_graph_activation \
      "電車には乗れていますか？" caps5_graph_structure_eval_v3/figures/graph_activation_ptsd.png

左：恐怖記憶グラフ全体（約470ノード）。右：同じ配置のまま、その発話で活性化したノードだけを
色付き＋名前ラベルで浮かび上がらせ、他は薄いグレーに落とす。「電車/車内…がこの大きな網の
一部（ノード）だ」を一目で伝えるための図。OpenAI に実接続（発話の埋め込み）。
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = [
    "Hiragino Sans", "Hiragino Maru Gothic Pro", "YuGothic", "AppleGothic", "sans-serif"]
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from actr_foa_kozak_v2 import config
from actr_foa_kozak_v2.infrastructure.embeddings import EmbeddingService
from actr_foa_kozak_v2.retrieval.spreading import SpreadingActivation

from ..pipeline import snapshot

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GRAPH = os.path.join(_HERE, "data", "graph_exp_ptsd.json")

COLORS = {
    config.STIMULUS_LABEL: "#f58231",
    config.RESPONSE_LABEL: "#4363d8",
    config.MEANING_LABEL: "#e6194B",
    config.EPISODE_LABEL: "#2E8B3D",
}
LEGEND = [("#f58231", "刺激"), ("#4363d8", "反応"), ("#e6194B", "意味"), ("#2E8B3D", "記憶")]


def _build(graph):
    G = nx.Graph()
    color, label, name = {}, {}, {}
    for n in graph.nodes:
        G.add_node(n.key); color[n.key] = COLORS.get(n.label, "#999")
        label[n.key] = n.label; name[n.key] = n.name
    for e in graph.episodes:
        G.add_node(e.key); color[e.key] = COLORS[config.EPISODE_LABEL]
        label[e.key] = config.EPISODE_LABEL; name[e.key] = e.event
    for ed in graph.edges:
        if ed.src in color and ed.dst in color:
            G.add_edge(ed.src, ed.dst)
    return G, color, label, name


def render(query: str, graph_path: str, out: str) -> None:
    graph = snapshot.load(graph_path)
    G, color, label, name = _build(graph)
    print(f"  layout: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges ...")
    pos = nx.spring_layout(G, k=0.32, iterations=70, seed=42)

    spreader = SpreadingActivation(EmbeddingService())
    activation, _reached, seeds = spreader.compute(query, graph)
    seed_keys = {s.key for s, _ in seeds}
    allact = {k: v for k, v in activation.items() if k in color}
    amax = max(allact.values()) if allact else 1.0
    # 「強く」活性化した上位だけを色付きにする（弱い波及はグレーに落として"一部感"を出す）。
    # 上位固定にして、刺激・反応・意味・記憶の4種が入るようにする。
    ranked = sorted(allact.items(), key=lambda x: x[1], reverse=True)
    hot_keys = set(seed_keys) | {k for k, _ in ranked[:16]}
    act = {k: allact[k] for k in hot_keys}
    lit = sorted(act.items(), key=lambda x: x[1], reverse=True)
    # 名前ラベルは入口＋活性上位の少数だけ（重なり回避）
    label_keys = set(seed_keys) | {k for k, _ in lit[:4]}

    fig, axes = plt.subplots(1, 2, figsize=(17, 8.6))

    # ---- 左：全体 ----
    axL = axes[0]
    nx.draw_networkx_edges(G, pos, ax=axL, edge_color="#cdd2d6", width=0.35, alpha=0.45)
    nx.draw_networkx_nodes(G, pos, ax=axL, node_color=[color[n] for n in G.nodes()],
                           node_size=26, alpha=0.9, linewidths=0)
    axL.set_title(f"① 恐怖記憶グラフ全体（{G.number_of_nodes()}ノード）",
                  fontsize=15, color="#26323B", fontweight="bold")
    axL.axis("off")
    axL.legend(handles=[mpatches.Patch(color=c, label=l) for c, l in LEGEND],
               loc="lower left", frameon=False, fontsize=12)

    # ---- 右：一部だけ活性化 ----
    axR = axes[1]
    nx.draw_networkx_edges(G, pos, ax=axR, edge_color="#e3e6e8", width=0.3, alpha=0.35)
    # 非活性＝薄いグレーで小さく
    dim = [n for n in G.nodes() if n not in act]
    nx.draw_networkx_nodes(G, pos, ax=axR, nodelist=dim, node_color="#DDE1E3",
                           node_size=14, alpha=0.55, linewidths=0)
    # 活性＝色付き・活性に比例したサイズ・入口は黒縁
    hot = [n for n in G.nodes() if n in act]
    sizes = [80 + 620 * (act[n] / amax) for n in hot]
    edgec = ["#111" if n in seed_keys else "none" for n in hot]
    nx.draw_networkx_nodes(G, pos, ax=axR, nodelist=hot, node_color=[color[n] for n in hot],
                           node_size=sizes, alpha=0.96,
                           edgecolors=edgec, linewidths=[1.8 if n in seed_keys else 0 for n in hot])
    labels = {k: (name[k][:8]) for k in label_keys if k in G.nodes()}
    # ラベルはノードの少し上に置いて重なりを減らす
    lpos = {k: (x, y + 0.028) for k, (x, y) in pos.items()}
    nx.draw_networkx_labels(G, lpos, ax=axR, labels=labels, font_size=10.5,
                            font_family=matplotlib.rcParams["font.family"][0],
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    axR.set_title(f"② 「{query}」で強く活性化した点（黒縁＝入口／{len(hot)}点）",
                  fontsize=15, color="#26323B", fontweight="bold")
    axR.axis("off")

    fig.suptitle("発話は、大きな記憶ネットワークの“ごく一部”だけを活性化する",
                 fontsize=16.5, color="#26323B", fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=145, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ {out}  （活性ノード {len(hot)} / 全 {G.number_of_nodes()}）")


if __name__ == "__main__":
    config.load_env()
    q = sys.argv[1] if len(sys.argv) > 1 else "電車には乗れていますか？"
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_HERE, "figures", "graph_activation_ptsd.png")
    render(q, DEFAULT_GRAPH, out)
