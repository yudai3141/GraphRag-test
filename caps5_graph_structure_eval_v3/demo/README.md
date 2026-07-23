# demo — 発話 → 抽出 → 応答を目で確かめる

臨床家が話しかけると、恐怖記憶グラフの中で**何が取り出され（入口の刺激→からだの反応→よぎった意味→
思い出した過去）、どんな発話が出るか**を可視化する。prompt で症状を指定するのではなく、同じ1つの
グラフに発話を入れ、引き金が立てば破局モード、空振りなら平静モードになる様子を観察できる。

Neo4j 非依存（スナップショット JSON からグラフを読む）。応答生成・拡散活性・引き金ゲートは
`actr_foa_kozak_v2` の実装をそのまま流用する。実行には OpenAI への実接続（埋め込み＋応答生成）が要る。

## ファイル

| ファイル | 役割 |
|---|---|
| `probe.py` | コア。`probe(query, graph, spreader, llm)` が1発話ぶんのトレース（入口/反応/意味/過去/モード/応答）を返す。臨床面接の既定バッテリー `DEFAULT_BATTERY` も持つ |
| `render_demo.py` | バッテリーを通し、**自己完結 HTML と Markdown**（質問応答例）を書き出すバッチ |
| `app.py` | **ライブ UI**（Streamlit）。発話を打つとその場で内部連想と応答が出る |

## 使い方

```bash
# ① 質問応答例を一括生成（HTML + Markdown）。グラフは差し替え可
uv run python -m caps5_graph_structure_eval_v3.demo.render_demo \
    caps5_graph_structure_eval_v3/data/graph_exp_ptsd.json
#   → caps5_graph_structure_eval_v3/demo/qa_examples.html / .md

# ② ライブで話しかけて観察
uv run --extra ui streamlit run caps5_graph_structure_eval_v3/demo/app.py
```

`data/graph_exp_{ptsd,balanced,healthy}.json` を差し替えると、同じ発話に対する反応の違い
（トラウマのみ vs 健常）を比べられる。グラフ JSON は大きく gitignore なので、無ければ
`pipeline/build_snapshot.py` で再構築する（→ 親フォルダの README）。

## 生成物

- `qa_examples.html` / `qa_examples.md` … 既定バッテリーの実出力（応答は temperature>0 なので再生成で変わる）。
  臨床監修 MTG 用のたたき台（`../meetings/` の観点 D「面接ダイアログの妥当性」に対応）。

> 注意：入口ステップは埋め込み依存なので、「話すこと」「夜」のような一般語にも着火し、挨拶でも
> 破局モードに入ることがある。これは既知の限界（論文 Limitations）であり、面接としての妥当性
> （reality break）を臨床監修に確認する材料そのものになる。
