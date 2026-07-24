"""ライブ確認 UI：発話を打ち込むと、そこから何が取り出され、どんな発話が出るかをその場で見る。

  uv run --extra ui streamlit run caps5_graph_structure_eval_v3/demo/app.py

Neo4j 非依存（スナップショット JSON）。左に会話、右に「入口の刺激→反応→意味→思い出した過去」の
内部連想と、平静／破局モードの判定を出す。グラフは読み取り専用（v2 Stage 1）。
"""

import os
import sys

# `streamlit run app.py` はこのファイルをパッケージ外のスクリプトとして実行するため、
# リポジトリルートを import パスに通し、絶対 import で読む（相対 import は使えない）。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st

from actr_foa_kozak_v2 import config
from actr_foa_kozak_v2.infrastructure.embeddings import EmbeddingService
from actr_foa_kozak_v2.infrastructure.llm import LLMClient
from actr_foa_kozak_v2.retrieval.spreading import SpreadingActivation

from caps5_graph_structure_eval_v3.demo import probe
from caps5_graph_structure_eval_v3.pipeline import snapshot

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GRAPH = os.path.join(_HERE, "data", "graph_exp_ptsd.json")


@st.cache_resource
def _load(graph_path: str):
    config.load_env()
    graph = snapshot.load(graph_path)
    spreader = SpreadingActivation(EmbeddingService())
    llm = LLMClient()
    return graph, spreader, llm


def _list(items, kind):
    if not items:
        st.caption("—")
        return
    for it in items:
        if kind == "episode":
            name, val, act = it
            st.markdown(f"- {name[:48]}  \n  <small>{val} · {act}</small>", unsafe_allow_html=True)
        else:
            name, act = it
            st.markdown(f"- {name}  <small>({act})</small>", unsafe_allow_html=True)


st.set_page_config(page_title="模擬患者デモ：発話→抽出→応答", layout="wide")
st.title("発話を入れると、何が取り出され、何を言うか")

graph_path = st.sidebar.text_input("グラフ(JSON)", DEFAULT_GRAPH)
st.sidebar.caption("トラウマのみ／balanced／健常を差し替えて挙動を比べられる。")

graph, spreader, llm = _load(graph_path)
st.sidebar.success(f"点 {len(graph.nodes)} / 記憶 {len(graph.episodes)} / 線 {len(graph.edges)}")

query = st.chat_input("臨床家として話しかける（例：夜は眠れていますか？）")
if query:
    trace = probe.probe(query, graph, spreader, llm)
    left, right = st.columns([1.1, 1])
    with left:
        st.chat_message("user").write(query)
        badge = "🔴 破局モード（引き金あり）" if trace["triggered"] else "🟦 平静モード（空振り）"
        st.chat_message("assistant").write(f"**{badge}**\n\n{trace['response']}")
    with right:
        st.markdown("#### 内部で取り出されたもの")
        st.markdown("**入口の刺激**"); _list(trace["seeds"], "stim")
        st.markdown("**からだの反応**"); _list(trace["responses"], "resp")
        st.markdown("**よぎった意味**"); _list(trace["meanings"], "mean")
        st.markdown("**思い出した過去**"); _list(trace["episodes"], "episode")
