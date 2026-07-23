"""発話バッテリーを恐怖記憶グラフに通し、「入力→抽出→応答」を自己完結 HTML と Markdown に書き出す。

  uv run python -m caps5_graph_structure_eval_v3.demo.render_demo \
      caps5_graph_structure_eval_v3/data/graph_exp_ptsd.json

各発話ごとに、平静／破局のどちらのモードに入ったか、入口の刺激・からだの反応・よぎった意味・
思い出した過去、そして実際に生成された発話を1枚のカードで見せる。OpenAI に実接続する
（埋め込み＋応答生成）。応答生成に失敗しても抽出結果だけは表示する。
"""

import html
import os
import sys
from typing import List

from actr_foa_kozak_v2 import config
from actr_foa_kozak_v2.infrastructure.embeddings import EmbeddingService
from actr_foa_kozak_v2.retrieval.spreading import SpreadingActivation

from . import probe
from ..pipeline import snapshot

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GRAPH = os.path.join(_HERE, "data", "graph_exp_ptsd.json")
DEFAULT_OUT = os.path.join(_HERE, "demo", "qa_examples")


def _chips(items, kind: str) -> str:
    if not items:
        return '<span class="empty">—</span>'
    out = []
    for it in items:
        if kind == "episode":
            name, val, act = it
            out.append(f'<span class="chip epi {val}">{html.escape(name[:40])}'
                       f'<span class="v">{val}·{act}</span></span>')
        else:
            name, act = it
            out.append(f'<span class="chip {kind}">{html.escape(str(name)[:34])}'
                       f'<span class="v">{act}</span></span>')
    return "".join(out)


def _card(t: dict) -> str:
    mode = ("破局モード（引き金あり）", "leap") if t["triggered"] else ("平静モード（空振り）", "calm")
    resp = t["response"] or '<span class="empty">（応答生成なし／失敗）</span>'
    return f"""
<div class="card {mode[1]}">
  <div class="q">Q. {html.escape(t['query'])}</div>
  <div class="mode {mode[1]}">{mode[0]}</div>
  <div class="grid">
    <div class="row"><span class="lab stim">入口の刺激</span><span class="vals">{_chips(t['seeds'],'stim')}</span></div>
    <div class="row"><span class="lab resp">からだの反応</span><span class="vals">{_chips(t['responses'],'resp')}</span></div>
    <div class="row"><span class="lab mean">よぎった意味</span><span class="vals">{_chips(t['meanings'],'mean')}</span></div>
    <div class="row"><span class="lab epi">思い出した過去</span><span class="vals">{_chips(t['episodes'],'episode')}</span></div>
  </div>
  <div class="ans"><span class="albl">生成された発話</span><p>{html.escape(t['response']) if t['response'] else resp}</p></div>
</div>"""


_CSS = """
:root{--ink:#26323B;--muted:#5C6E79;--paper:#FAFBFA;--card:#fff;--line:#E2E7E5;--accent:#0E7490;--soft:#EAF4F6;
--stim:#f58231;--resp:#4363d8;--mean:#e6194B;--epi:#2E8B3D;--leap:#B4325A;--calm:#0E7490;}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);
font-family:"Hiragino Sans","Hiragino Kaku Gothic ProN",sans-serif;font-size:15px;line-height:1.85}
.wrap{max-width:840px;margin:0 auto;padding:52px 22px 90px}
.eyebrow{color:var(--accent);font-size:12.5px;font-weight:600;letter-spacing:.13em;margin:0 0 8px}
h1{font-family:"Hiragino Mincho ProN","Yu Mincho",serif;font-size:25px;font-weight:600;margin:0 0 10px}
.lede{color:var(--muted);max-width:44em;margin:0 0 8px}
.meta{color:var(--muted);font-size:12.5px;margin:6px 0 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px 22px;margin:22px 0;border-left:4px solid var(--line)}
.card.leap{border-left-color:var(--leap)}.card.calm{border-left-color:var(--calm)}
.q{font-size:16px;font-weight:700;margin-bottom:8px}
.mode{display:inline-block;font-size:11.5px;font-weight:700;border-radius:20px;padding:2px 12px;margin-bottom:14px}
.mode.leap{background:#FBEAF0;color:var(--leap)}.mode.calm{background:var(--soft);color:var(--calm)}
.grid{display:flex;flex-direction:column;gap:7px;margin:6px 0 14px}
.row{display:flex;gap:10px;align-items:flex-start}
.lab{flex:none;width:96px;font-size:12px;font-weight:700;padding-top:3px}
.lab.stim{color:var(--stim)}.lab.resp{color:var(--resp)}.lab.mean{color:var(--mean)}.lab.epi{color:var(--epi)}
.vals{display:flex;flex-wrap:wrap;gap:6px}
.chip{display:inline-flex;align-items:center;gap:6px;font-size:12.5px;border-radius:7px;padding:3px 9px;background:#F3F5F5;border:1px solid var(--line)}
.chip.stim{background:#FDEFE3}.chip.resp{background:#E9EDFB}.chip.mean{background:#FCE8ED}
.chip.epi.negative{background:#FBEAF0}.chip.epi.positive{background:#E8F6EA}.chip.epi.neutral{background:#F1F3F3}
.chip .v{font-size:10.5px;color:var(--muted)}
.empty{color:#AAB4B8}
.ans{background:#FBFCFC;border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin-top:6px}
.albl{font-size:11.5px;font-weight:700;color:var(--muted)}
.ans p{margin:6px 0 0;font-size:15.5px;line-height:1.9}
.legend{font-size:12.5px;color:var(--muted);background:var(--soft);border-radius:10px;padding:12px 16px;margin:20px 0}
footer{margin-top:60px;padding-top:18px;border-top:1px solid var(--line);font-size:12.5px;color:var(--muted)}
"""


def render_html(traces: List[dict], graph_name: str) -> str:
    cards = "".join(_card(t) for t in traces)
    n_leap = sum(1 for t in traces if t["triggered"])
    return f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>模擬患者の質問応答例 — 発話から何が取り出され、何を言うか</title>
<style>{_CSS}</style></head><body><div class="wrap">
<p class="eyebrow">デモ：発話 → 抽出 → 応答</p>
<h1>模擬患者は、質問から何を取り出して何を言うか</h1>
<p class="lede">臨床家が投げそうな発話を恐怖記憶グラフ（{html.escape(graph_name)}）に通し、内部で活性化した
刺激・反応・意味・過去の記憶と、そこから生成された発話を並べた実出力。prompt で症状を指定しては
いない——同じ1つのグラフに発話を入れ、引き金が立てば破局モード、空振りなら平静モードになる。</p>
<p class="meta">全 {len(traces)} 発話中 {n_leap} 件で引き金が立った（破局モード）。</p>
<div class="legend"><b>見方：</b>入口の刺激（橙）に発話が着火 → からだの反応（青）・よぎった意味（赤）へ拡散 →
思い出した過去（緑・valence 付き）。数値は活性の強さ。応答は内部連想を《口に出さない手がかり》として
生成したもの（途中経過は発話に出さない設計）。</div>
{cards}
<footer>caps5_graph_structure_eval_v3/demo/render_demo.py で生成。再生成すると応答は変わりうる（temperature&gt;0）。
グラフ: {html.escape(os.path.basename(graph_name))}</footer>
</div></body></html>"""


def render_markdown(traces: List[dict], graph_name: str) -> str:
    lines = [f"# 模擬患者の質問応答例（{graph_name}）\n",
             "臨床家の発話 → 内部で取り出されたもの → 生成された発話。\n"]
    for t in traces:
        mode = "破局モード（引き金あり）" if t["triggered"] else "平静モード（空振り）"
        lines.append(f"## Q. {t['query']}\n")
        lines.append(f"- **モード**: {mode}")
        lines.append(f"- **入口の刺激**: {', '.join(f'{n}({a})' for n,a in t['seeds']) or '—'}")
        lines.append(f"- **からだの反応**: {', '.join(f'{n}({a})' for n,a in t['responses']) or '—'}")
        lines.append(f"- **よぎった意味**: {', '.join(f'{n}({a})' for n,a in t['meanings']) or '—'}")
        lines.append(f"- **思い出した過去**: {', '.join(f'{e}[{v}·{a}]' for e,v,a in t['episodes']) or '—'}")
        lines.append(f"\n> {t['response'] or '（応答生成なし／失敗）'}\n")
    return "\n".join(lines)


def main() -> None:
    config.load_env()
    graph_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GRAPH
    out_base = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT

    graph = snapshot.load(graph_path)
    embedder = EmbeddingService()
    spreader = SpreadingActivation(embedder)

    try:
        from actr_foa_kozak_v2.infrastructure.llm import LLMClient
        llm = LLMClient()
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ LLM 初期化に失敗（応答生成なしで続行）: {e}")
        llm = None

    traces = []
    for i, q in enumerate(probe.DEFAULT_BATTERY, 1):
        try:
            t = probe.probe(q, graph, spreader, llm)
        except Exception as e:  # noqa: BLE001
            print(f"  ! 発話{i} で応答生成に失敗: {e}")
            t = probe.probe(q, graph, spreader, None)
        mode = "破局" if t["triggered"] else "平静"
        print(f"  {i}/{len(probe.DEFAULT_BATTERY)} [{mode}] {q[:24]}…")
        traces.append(t)

    name = os.path.basename(graph_path)
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    with open(out_base + ".html", "w") as f:
        f.write(render_html(traces, name))
    with open(out_base + ".md", "w") as f:
        f.write(render_markdown(traces, name))
    print(f"✅ {out_base}.html / .md を書き出し")


if __name__ == "__main__":
    main()
