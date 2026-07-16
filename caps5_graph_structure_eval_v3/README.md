# caps5_graph_structure_eval_v3 — 恐怖記憶グラフの定量評価（実験1）と論文

`actr_foa_kozak_v2` の恐怖記憶グラフ（拡散活性）を「測定対象」として、その妥当性を
定量評価するフォルダ。トラウマとの近さで層化した発話を入力し、CAPS-5 の侵入症状（B 基準）
に対応づけた指標で、グラフが「トラウマに関連するほど強く、無関係には弱く」反応するか
（過般化するか）を、同規模の健常統制と比べて示す。

**成果物（提出用）**：[`paper/short_paper_ja.pdf`](paper/short_paper_ja.pdf)（日本語）／
[`paper/short_paper.pdf`](paper/short_paper.pdf)（英語）。入口の図解は [`docs/00_overview.html`](docs/00_overview.html)。

## 二層オントロジー（理論忠実版）

この検討過程で `actr_foa_kozak_v2` のオントロジーを一次理論に忠実化した
（中核評価の固定4種・単一ハブ・外付けバイアス・関係別重み・負→負エッジを撤去）。

- **恐怖構造層**（Foa/Lang）：刺激 / 反応 / 意味（開語彙）＋ `EVOKES / MEANS / CO_OCCURS / SIMILAR`
- **エピソード記憶層**（二重表象理論）：エピソード ＋ 層間の橋 `RECALLS / BINDS`（記憶＝束ねられた刺激-反応-意味）
- 伝搬は無向・対称、重みはほぼ一様。怖がり度＝反応要素の活性。
- 経緯・設計・結果の詳細は `docs/` 各 HTML（`20260713_theory_audit` / `_ontology_redesign` /
  `_faithful_results` / `_corpus_sensitivity` / `20260715_experiment2_design`）。

## ファイルマップ

| | ファイル | 役割 |
|---|---|---|
| **パイプライン** | `pipeline/build_snapshot.py` | 記述テキスト → 恐怖記憶グラフ(JSON)。Neo4j 非依存 |
| | `pipeline/generate_healthy.py` | 健常統制用の日常記述を LLM 生成 |
| | `pipeline/cue_battery.py` | きっかけ発話を5レベル層化で生成（測定は L1–L4 を使用） |
| | `pipeline/runner.py` | グラフ × 条件で拡散活性を回し、指標を CSV 記録 |
| | `pipeline/analyze.py` | 集計＋図 `fig1–3` 生成＋ `results/summary.json` |
| | `pipeline/graph_image.py` | 実グラフの静止 PNG 描画 |
| | `pipeline/snapshot.py` | Neo4j ↔ JSON（旧経路・任意） |
| | `pipeline/viz_snapshot.py` | pyvis インタラクティブ HTML（任意） |
| **報告書** | `report/render_paper.py` | Markdown → 図埋め込み自己完結 HTML |
| | `report/render_pdf.sh` | HTML → PDF（ヘッドレス Chrome） |
| **成果物** | `paper/short_paper*.{md,html,pdf}` | ショートペーパー（日本語＝提出用／英語） |
| | `figures/` | 図（`fig1–3.png`・模式図 `*.svg`・実グラフ `graph_ptsd_view.png`） |
| | `results/` | `metrics_{ptsd,balanced,healthy}.csv`・`summary.json` |
| | `data/` | `cue_battery.json`・source 記述・`graph_exp_*.json`（大きい・gitignore） |
| | `docs/` | 設計・検討の図解 HTML 群 |
| | `PLAN.md` | 計画・決定事項ログ |

## 再現手順（リポジトリルートから）

要 `.env`（OpenAI）。グラフのスナップショットは大きく gitignore なので、まず再構築する。

```bash
# 1. 3グラフを同一パイプラインで構築（要 OpenAI）
uv run python -m caps5_graph_structure_eval_v3.pipeline.build_snapshot simulation/ptsd_story.txt \
    caps5_graph_structure_eval_v3/data/graph_exp_ptsd.json
uv run python -m caps5_graph_structure_eval_v3.pipeline.build_snapshot \
    simulation/ptsd_story.txt,caps5_graph_structure_eval_v3/data/daily_life_supplement.txt \
    caps5_graph_structure_eval_v3/data/graph_exp_balanced.json
uv run python -m caps5_graph_structure_eval_v3.pipeline.generate_healthy          # → data/healthy_narrative_long.txt
uv run python -m caps5_graph_structure_eval_v3.pipeline.build_snapshot caps5_graph_structure_eval_v3/data/healthy_narrative_long.txt \
    caps5_graph_structure_eval_v3/data/graph_exp_healthy.json

# 2. 各グラフを同じ刺激バッテリーで測定
for g in ptsd balanced healthy; do
  uv run python -m caps5_graph_structure_eval_v3.pipeline.runner caps5_graph_structure_eval_v3/data/graph_exp_$g.json \
      caps5_graph_structure_eval_v3/data/cue_battery.json caps5_graph_structure_eval_v3/results/metrics_$g.csv
done

# 3. 集計＋図（fig1–3, summary.json）
uv run --with matplotlib python -m caps5_graph_structure_eval_v3.pipeline.analyze
# 4. 実グラフの静止画
uv run --with matplotlib --with networkx python -m caps5_graph_structure_eval_v3.pipeline.graph_image \
    caps5_graph_structure_eval_v3/data/graph_exp_ptsd.json caps5_graph_structure_eval_v3/figures/graph_ptsd_view.png "PTSD graph"
# 5. 報告書 HTML → PDF
uv run --with markdown python -m caps5_graph_structure_eval_v3.report.render_paper short_paper
uv run --with markdown python -m caps5_graph_structure_eval_v3.report.render_paper short_paper_ja
bash caps5_graph_structure_eval_v3/report/render_pdf.sh
```

## 主な結果

- **一機構・三読み出しの分離**：同じ拡散活性を反応/意味/エピソードで読むと B5/B4/B1 の別々の勾配。いずれも L2/L3 で高止まり＝過般化。
- **機構の必要性**：`no_spread` で反応 0.00（伝搬が怖がりを生む）／`no_bind` で侵入記憶が約半減、反応は不変（束ねは記憶想起に特異）。
- **健常統制との対比**：PTSD は曖昧なきっかけでも 75%、健常は中立で 5%。過般化は手法ではなくトラウマ記憶由来。

## 今後

- B3（フラッシュバック）＝ S-rep/C-rep 分割による `S/(S+C)` の実装・測定
- 回復動態（残留活性の減衰）／入口マッチングの改善
- 実験2：CAPS-5 に基づく定量評価（臨床家が採点枠組みを監査）
