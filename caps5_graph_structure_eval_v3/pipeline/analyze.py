"""実験1の解析＋図生成。metrics CSV（3グラフ）を読み、指標を集計し PNG 図と summary.json を出す。

実行: uv run --with matplotlib python -m caps5_graph_structure_eval_v3.analyze
前提: results/metrics_{ptsd,balanced,healthy}.csv が存在すること（runner を各グラフで実行）。
"""

import csv
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(_HERE, "results")
FIG = os.path.join(_HERE, "figures")
os.makedirs(FIG, exist_ok=True)

GRAPHS = {
    "PTSD (trauma-only)": "metrics_ptsd.csv",
    "PTSD (balanced)": "metrics_balanced.csv",
    "Healthy control": "metrics_healthy.csv",
}
# 図は L1〜L4（トラウマ→中立の過般化勾配）に絞る。L5(positive)は入口アーティファクトのため図から除外（データは保持）。
LEVELS = ["1", "2", "3", "4"]
LEVEL_LABELS = ["L1\ntrauma", "L2\nrelated", "L3\nambiguous", "L4\nneutral"]

INK = "#26323B"; MUT = "#5C6E79"; GRID = "#E2E7E5"
C_PTSD = "#B4325A"; C_HEALTHY = "#0C8CA8"; C_TRAUMA = "#B26A00"; C_GRAY = "#98A4AC"


def load(fname):
    p = os.path.join(RES, fname)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def mean_at(rows, cond, metric, level):
    xs = [float(r[metric]) for r in rows if r["condition"] == cond and r["level"] == level]
    return st.mean(xs) if xs else 0.0


def rate_at(rows, cond, level):
    xs = [int(r["triggered"]) for r in rows if r["condition"] == cond and r["level"] == level]
    return (sum(xs) / len(xs)) if xs else 0.0


def conc_at(rows, cond, level):
    xs = []
    for r in rows:
        if r["condition"] == cond and r["level"] == level:
            t = float(r["total_act"])
            xs.append(float(r["neg_ep_act"]) / t if t > 0 else 0.0)
    return st.mean(xs) if xs else 0.0


def style_ax(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUT, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    data = {name: load(f) for name, f in GRAPHS.items()}
    for name, rows in data.items():
        print(f"{name}: {len(rows)} rows" + ("" if rows else "  ← MISSING"))

    summary = {"per_graph": {}, "readouts": {}}

    # ---- collect per-graph, faithful, per-level R/M/E/trigger/conc ----
    METRICS = {"resp_act": "B5 physiological (R)", "meaning_act": "B4 distress (M)",
               "neg_ep_act": "B1 intrusion (E)"}
    for name, rows in data.items():
        if not rows:
            continue
        g = {"valence_note": ""}
        for m in METRICS:
            g[m] = [mean_at(rows, "faithful", m, lv) for lv in LEVELS]
        g["trigger"] = [rate_at(rows, "faithful", lv) for lv in LEVELS]
        g["conc"] = [conc_at(rows, "faithful", lv) for lv in LEVELS]
        # gradients L1/L4
        for m in METRICS:
            v = g[m]
            g[f"{m}_L1_L4"] = (v[0] / v[3]) if v[3] > 1e-9 else float("inf")
        summary["per_graph"][name] = g

    # ===== FIG 1: dissociable readouts within the PTSD graph (magnitudes, one graph) =====
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    style_ax(ax)
    po = summary["per_graph"].get("PTSD (trauma-only)", {})
    rd = {"resp_act": ("B5 — how much the bodily-response nodes fire", C_PTSD, "-o"),
          "meaning_act": ("B4 — how much the threat-meaning nodes fire", C_TRAUMA, "-s"),
          "neg_ep_act": ("B1 — how much the negative-memory nodes fire", C_HEALTHY, "-^")}
    for m, (title, col, mk) in rd.items():
        v = po.get(m, [0]*len(LEVELS))
        ax.plot(range(len(LEVELS)), v, mk, color=col, lw=2.2, ms=7, label=title)
    ax.set_xticks(range(len(LEVELS))); ax.set_xticklabels(LEVEL_LABELS, fontsize=9)
    ax.set_ylabel("activation of those nodes\n(relative units; higher = stronger reaction)", color=MUT, fontsize=9.5)
    ax.set_xlabel("how trauma-related the cue is", color=MUT, fontsize=9.5)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    ax.set_title("Fig. 1  How strongly each symptom's nodes light up, by cue type (PTSD graph)",
                 fontsize=11.5, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_readouts.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ================= FIG 2: mechanism ablation (PTSD trauma-only) =================
    # no_spread(=0) と no_similar(fan集中で逆に増える)は図から除外し本文で言及。
    # クリーンに解釈できる faithful vs no-BINDS（束ねの寄与）だけを示す。
    bal = data.get("PTSD (trauma-only)", [])
    conds = ["faithful", "no_bind"]
    cond_labels = ["faithful\n(full model)", "no-BINDS\n(unbind memory)"]
    summary["no_spread_L1"] = {"resp_act": mean_at(bal, "no_spread", "resp_act", "1"),
                               "neg_ep_act": mean_at(bal, "no_spread", "neg_ep_act", "1")}
    summary["no_similar_L1"] = {"resp_act": mean_at(bal, "no_similar", "resp_act", "1"),
                                "neg_ep_act": mean_at(bal, "no_similar", "neg_ep_act", "1")}
    respL1 = [mean_at(bal, c, "resp_act", "1") for c in conds]
    epL1 = [mean_at(bal, c, "neg_ep_act", "1") for c in conds]
    summary["ablation_L1"] = {"conditions": conds, "resp_act": respL1, "neg_ep_act": epL1}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.8))
    for ax, vals, ttl, col in zip(axes, [respL1, epL1],
                                  ["Bodily response (R)\nat trauma cues", "Memory recall (E)\nat trauma cues"],
                                  [C_PTSD, C_TRAUMA]):
        style_ax(ax)
        bars = ax.bar(range(len(conds)), vals, color=col, width=0.55)
        ax.set_title(ttl, fontsize=10.5, color=INK, fontweight="bold")
        ax.set_xticks(range(len(conds))); ax.set_xticklabels(cond_labels, fontsize=9)
        ax.set_ylim(bottom=0)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=INK)
    fig.suptitle("Fig. 2  Episode binding drives intrusive-memory recall (ablations)",
                 fontsize=11.5, color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_ablation.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ================= FIG 3: corpus & control — trigger rate by level =================
    fig, ax = plt.subplots(figsize=(8, 4.2))
    style_ax(ax)
    series = [("PTSD (trauma-only)", C_TRAUMA, "-o"),
              ("PTSD (balanced)", C_PTSD, "-o"),
              ("Healthy control", C_HEALTHY, "-s")]
    for name, col, mk in series:
        rows = data.get(name, [])
        if not rows:
            continue
        y = [rate_at(rows, "faithful", lv)*100 for lv in LEVELS]
        ax.plot(range(len(LEVELS)), y, mk, color=col, lw=2.2, ms=6, label=name)
        summary["per_graph"].setdefault(name, {})["trigger_pct"] = y
    ax.set_xticks(range(len(LEVELS))); ax.set_xticklabels(LEVEL_LABELS, fontsize=9)
    ax.set_ylabel("how often it reacts fearfully\n(% of the 20 cues at this level)", color=MUT, fontsize=9.5)
    ax.set_xlabel("how trauma-related the cue is", color=MUT, fontsize=9.5)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9.5)
    ax.set_title("Fig. 3  How often each graph reacts fearfully, by cue type",
                 fontsize=11.5, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_corpus_control.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---- print & save summary ----
    print("\n=== per-graph faithful gradients (L1..L5) ===")
    for name, g in summary["per_graph"].items():
        if "resp_act" not in g:
            continue
        print(f"\n{name}")
        for m, title in METRICS.items():
            v = g[m]
            print(f"  {title:22s} " + " ".join(f"{x:.3f}" for x in v) + f"   L1/L4={g[m+'_L1_L4']:.2f}")
        print(f"  trigger %             " + " ".join(f"{x*100:4.0f}" for x in g["trigger"]))

    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=str)
    print(f"\n✅ figures → {FIG}/  summary → {RES}/summary.json")


if __name__ == "__main__":
    main()
