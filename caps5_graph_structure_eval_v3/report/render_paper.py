"""short_paper.md → 自己完結HTML（図を base64 で埋め込み）。
実行: uv run --with markdown python -m caps5_graph_structure_eval_v3.render_paper
"""
import base64
import os
import re

import markdown

import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_name = sys.argv[1] if len(sys.argv) > 1 else "short_paper"
MD = os.path.join(_HERE, "paper", f"{_name}.md")
OUT = os.path.join(_HERE, "paper", f"{_name}.html")

text = open(MD).read()
html_body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])


def embed(m):
    src = m.group(1)
    path = os.path.normpath(os.path.join(_HERE, "paper", src))
    if os.path.exists(path):
        b64 = base64.b64encode(open(path, "rb").read()).decode()
        return f'src="data:image/png;base64,{b64}"'
    return m.group(0)


html_body = re.sub(r'src="([^"]+\.png)"', embed, html_body)


def inline_svg(m):
    src = m.group(1)
    path = os.path.normpath(os.path.join(_HERE, "paper", src))
    if os.path.exists(path):
        return f'<div class="schematic">{open(path).read()}</div>'
    return m.group(0)


# markdown turns ![](x.svg) into <img ... src=".../x.svg" ...>; inline the SVG file content.
html_body = re.sub(r'<img[^>]*src="([^"]+\.svg)"[^>]*>', inline_svg, html_body)

STYLE = """
body{font-family:"Hiragino Mincho ProN","Times New Roman",Georgia,serif;color:#1a1a1a;
  max-width:820px;margin:0 auto;padding:48px 28px 90px;line-height:1.65;font-size:16px;background:#fff}
h1{font-size:25px;line-height:1.3;text-align:center;margin:0 0 6px}
h2{font-size:19px;border-bottom:1px solid #ddd;padding-bottom:4px;margin-top:38px}
h3{font-size:16px;margin-top:26px}
p{margin:11px 0}
em{color:#444}
img{display:block;max-width:100%;margin:18px auto;border:1px solid #eee;border-radius:4px}
table{border-collapse:collapse;margin:18px auto;font-size:14px;font-family:"Helvetica Neue",Arial,sans-serif}
th,td{border:1px solid #ccc;padding:6px 12px;text-align:left}
th{background:#f4f5f6}
code{background:#f2f3f4;padding:1px 5px;border-radius:3px;font-size:13px}
ul{font-size:14.5px}
a{color:#0E7490}
.meta{text-align:center;color:#666;font-size:13px;font-style:italic;margin-bottom:24px}
blockquote{background:#EAF4F6;border-left:4px solid #0E7490;margin:18px 0;padding:11px 18px;border-radius:0 6px 6px 0;font-size:15px;color:#15343d;font-family:"Helvetica Neue",Arial,sans-serif}
blockquote p{margin:4px 0}
blockquote strong{color:#0E7490}
em{color:#333}
.schematic{margin:18px auto;text-align:center}
.schematic svg{max-width:100%;height:auto;border:1px solid #eee;border-radius:4px;background:#fff;padding:6px}
.figcap{font-size:13px;color:#555;text-align:center;margin:-8px 0 18px;font-style:italic}
"""
html = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>Symptoms from Structure — short paper</title><style>{STYLE}</style></head>'
        f'<body>{html_body}</body></html>')
open(OUT, "w").write(html)
print(f"✅ {OUT}  ({len(html)//1024} KB, 図埋め込み済み)")
