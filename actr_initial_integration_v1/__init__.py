"""PTSD 対話エージェント（ACT-R 初期統合 v1）。

レイヤー構成:
  domain         … 純粋なデータ構造
  infrastructure … 外部I/O（Neo4j / 埋め込み / LLM）
  memory         … テキスト→構造化エピソードグラフの構築
  retrieval      … ACT-R activation による記憶検索
  prompts        … プロンプト
  agent          … 対話オーケストレーション
"""
