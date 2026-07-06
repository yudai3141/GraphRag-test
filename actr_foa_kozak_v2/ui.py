"""自由に話しかけて「連想」を観察する Web UI（Streamlit）。

  uv run --extra ui streamlit run actr_foa_kozak_v2/ui.py

左に会話、各応答の下に「内部の連想（字幕）」、右に恐怖構造グラフの活性ヒートマップを出す。
話しかけるたびに、恐怖構造のどこが光り、危険/無力の中核へ収束していくかが見える。
記憶（グラフ）は変更しない読み取り専用（＝v2 Stage 1）。
"""

import os
import sys

# streamlit run はスクリプト実行なので、リポジトリルートを import パスに通す
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import streamlit.components.v1 as components

from actr_foa_kozak_v2 import activation_viz, config
from actr_foa_kozak_v2.agent import debug_view
from actr_foa_kozak_v2.infrastructure.embeddings import EmbeddingService
from actr_foa_kozak_v2.infrastructure.llm import LLMClient
from actr_foa_kozak_v2.infrastructure.neo4j_store import Neo4jStore
from actr_foa_kozak_v2.prompts import templates
from actr_foa_kozak_v2.retrieval.spreading import SpreadingActivation


@st.cache_resource(show_spinner="恐怖構造グラフを読み込み中…")
def load():
    config.load_env()
    store = Neo4jStore()
    graph = store.load_fear_graph()
    store.close()
    embedder = EmbeddingService()
    return graph, SpreadingActivation(embedder), LLMClient()


st.set_page_config(page_title="PTSD 恐怖構造エージェント v2", layout="wide")
graph, spreader, llm = load()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.latest_html = None

with st.sidebar:
    st.markdown("### PTSD 恐怖構造エージェント v2")
    st.caption(
        f"刺激 {len(graph.stimuli())} / ノード {len(graph.nodes)} / "
        f"エピソード {len(graph.episodes)} / エッジ {len(graph.edges)}"
    )
    st.markdown(
        "**色**: 🟠刺激 🔵反応 🔴意味 🟣中核 🟢エピソード ⚪非活性\n\n"
        "話しかけると、内部で恐怖構造をたどった連想が起き、右のグラフが光ります。"
        "一見中立な話題でも、過去の嫌な記憶や『危険/無力』へ連想が伸びます。"
    )
    if st.button("会話をリセット"):
        st.session_state.messages = []
        st.session_state.latest_html = None
        st.rerun()

# --- 入力処理（この描画に反映させるため先に処理する） --------------------
prompt = st.chat_input("自由に話しかけてください…")
if prompt:
    result = spreader.recall(prompt, graph)
    activation, reached_hop, seeds = spreader.compute(prompt, graph)
    cap = debug_view.caption(result)
    response = llm.complete(templates.SYSTEM, templates.build_user_prompt(prompt, result))
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response, "caption": cap})
    st.session_state.latest_html = activation_viz.render_html(
        graph, activation, reached_hop, seeds, prompt, height="620px"
    )

# --- 表示 ---------------------------------------------------------------
col_chat, col_view = st.columns([5, 4], gap="large")

with col_chat:
    st.subheader("会話")
    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🫥"):
            st.markdown(m["content"])
            if m.get("caption"):
                with st.container(border=True):
                    st.markdown("**🧠 内部の連想（字幕）**")
                    st.markdown(m["caption"])
    if not st.session_state.messages:
        st.info("下の入力欄から自由に話しかけてください。例:「電車ってよく乗るの？」「勉強どう？」")

with col_view:
    st.subheader("恐怖構造の活性")
    if st.session_state.latest_html:
        components.html(st.session_state.latest_html, height=660, scrolling=False)
    else:
        st.info("話しかけると、恐怖構造のどこが活性化したかが光ります。")
