#!/bin/bash
# HTML報告書をヘッダ/フッタなしPDFに変換（ヘッドレスChrome）。
# 使い方: bash caps5_graph_structure_eval_v3/render_pdf.sh
set -e
cd "$(dirname "$0")/../.."
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
for name in short_paper short_paper_ja; do
  "$CHROME" --headless --disable-gpu --no-pdf-header-footer \
    --print-to-pdf="caps5_graph_structure_eval_v3/paper/$name.pdf" \
    "file://$PWD/caps5_graph_structure_eval_v3/paper/$name.html" >/dev/null 2>&1
  echo "✅ caps5_graph_structure_eval_v3/paper/$name.pdf"
done
