# -*- coding: utf-8 -*-
"""GraphRAG

LangChain + Neo4j を使った GraphRAG の実装。
元は Colab ノートブック (https://colab.research.google.com/github/nyanta012/demo/blob/main/GraphRAG.ipynb)。
VSCode 上で扱えるよう、各処理を関数化した。

参考: https://blog.langchain.dev/enhancing-rag-based-applications-accuracy-by-constructing-and-leveraging-knowledge-graphs/

セットアップ (uv 管理):
    uv sync                       # 依存関係をインストール
    uv run python graphrag.py     # 実行
    uv sync --extra viz           # show_graph() を使う場合 (yfiles)
"""

import os
from typing import List

from dotenv import load_dotenv

from neo4j import GraphDatabase

from pydantic import BaseModel, Field

from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import TokenTextSplitter

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer

from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars


def load_environment() -> None:
    """`.env` から環境変数を読み込む。

    必要なキー: OPENAI_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
    """
    load_dotenv()
    required = ["OPENAI_API_KEY", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise EnvironmentError(
            f".env に次の環境変数が設定されていません: {', '.join(missing)}"
        )


def build_llm(model_name: str = "gpt-5.4") -> ChatOpenAI:
    """グラフ構築・回答生成に使う LLM を返す。"""
    return ChatOpenAI(temperature=0, model_name=model_name)


def load_documents(file_path: str, chunk_size: int = 512, chunk_overlap: int = 125):
    """テキストファイルを読み込み、トークン単位で分割する。"""
    raw_documents = TextLoader(file_path).load()
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(raw_documents)


def build_graph(documents, llm: ChatOpenAI) -> Neo4jGraph:
    """ドキュメントから知識グラフを構築し、Neo4j に登録する。"""
    llm_transformer = LLMGraphTransformer(llm=llm)
    graph_documents = llm_transformer.convert_to_graph_documents(documents)

    graph = Neo4jGraph()
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True,
    )
    return graph


def show_graph(cypher: str = "MATCH (s)-[r:!MENTIONS]->(t) RETURN s,r,t LIMIT 50"):
    """Cypher クエリの結果をグラフウィジェットとして返す。

    注意: yfiles_jupyter_graphs を利用するため Jupyter / VSCode Notebook 上でのみ
    表示できる。通常の Python スクリプト実行では描画されない。
    """
    from yfiles_jupyter_graphs import GraphWidget

    driver = GraphDatabase.driver(
        uri=os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    session = driver.session(database=os.environ.get("NEO4J_DATABASE", "neo4j"))
    widget = GraphWidget(graph=session.run(cypher).graph())
    widget.node_label_mapping = "id"
    return widget


# 凡例・ノード色分けに使うカラーパレット（足りなければ循環して使う）
_PALETTE = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#ffe119",
]


def _node_group(node) -> str:
    """ノードの代表ラベルを返す（`__Entity__` などの内部ラベルは除外）。"""
    labels = [label for label in node.labels if not label.startswith("__")]
    return labels[0] if labels else "Node"


def export_graph_html(
    cypher: str = "MATCH (s)-[r:!MENTIONS]->(t) RETURN s,r,t LIMIT 50",
    output_path: str = "graph.html",
) -> str:
    """Cypher クエリの結果をインタラクティブな HTML として書き出す。

    ノードはラベル（種類）ごとに色分けし、画面右上に色の凡例を表示する。
    エッジにはリレーションの種類をラベルとして表示する。

    Jupyter を使わずに、通常の Python スクリプト実行でグラフを可視化できる。
    生成された HTML を VSCode の Simple Browser やブラウザで開くと、ノードを
    ドラッグできるインタラクティブなグラフが表示される。

    Returns:
        書き出した HTML ファイルのパス。
    """
    from pyvis.network import Network

    driver = GraphDatabase.driver(
        uri=os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    net = Network(height="750px", width="100%", directed=True, notebook=False)

    color_map: dict[str, str] = {}
    with driver.session(database=os.environ.get("NEO4J_DATABASE", "neo4j")) as session:
        result_graph = session.run(cypher).graph()
        for node in result_graph.nodes:
            group = _node_group(node)
            if group not in color_map:
                color_map[group] = _PALETTE[len(color_map) % len(_PALETTE)]
            label = node.get("id") or group
            title = f"[{group}]\n" + "\n".join(f"{k}: {v}" for k, v in node.items())
            net.add_node(
                node.element_id,
                label=str(label),
                title=title,
                shape="dot",
                color="#97c2fc",
                font={"color": color_map[group], "size": 16},
                group=group,
            )
        for rel in result_graph.relationships:
            net.add_edge(
                rel.start_node.element_id,
                rel.end_node.element_id,
                label=rel.type,
                title=rel.type,
                color="#9aa0a6",
                font={"size": 10, "color": "#555"},
            )
    driver.close()

    net.write_html(output_path, notebook=False)
    _inject_legend(output_path, color_map)
    print(f"Graph written to {output_path} ({len(color_map)} node types)")
    return output_path


def _inject_legend(html_path: str, color_map: dict) -> None:
    """生成済み HTML に、ノードの色凡例を右上に重ねて表示する。"""
    items = "".join(
        f'<div style="display:flex;align-items:center;margin:2px 0;">'
        f'<span style="display:inline-block;width:14px;height:14px;border-radius:3px;'
        f'background:{color};margin-right:6px;border:1px solid #0003;"></span>'
        f'<span>{group}</span></div>'
        for group, color in sorted(color_map.items())
    )
    legend = (
        '<div style="position:fixed;top:10px;right:10px;z-index:1000;'
        "background:rgba(255,255,255,0.95);border:1px solid #ccc;border-radius:8px;"
        "padding:10px 12px;font:12px/1.4 sans-serif;max-height:90vh;overflow:auto;"
        'box-shadow:0 2px 6px #0002;">'
        '<div style="font-weight:bold;margin-bottom:6px;">ノードの種類</div>'
        f"{items}"
        '<div style="margin-top:8px;color:#777;border-top:1px solid #eee;'
        'padding-top:6px;">エッジのラベル = 関係の種類</div>'
        "</div>"
    )
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace("</body>", legend + "</body>", 1)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(content)


def build_vector_index() -> Neo4jVector:
    """既存グラフからハイブリッド検索用のベクトルインデックスを作成する。"""
    return Neo4jVector.from_existing_graph(
        OpenAIEmbeddings(),
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding",
    )


# Extract entities from text
class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
        "appear in the text",
    )


def build_entity_chain(llm: ChatOpenAI):
    """テキストから人物・組織エンティティを抽出するチェーンを返す。"""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are extracting organization and person entities from the text.",
            ),
            (
                "human",
                "Use the given format to extract information from the following "
                "input: {question}",
            ),
        ]
    )
    return prompt | llm.with_structured_output(Entities)


def generate_full_text_query(input: str) -> str:
    """全文検索用のクエリ文字列を生成する。

    入力を単語に分割し、各単語に類似度しきい値 (~2 文字の変化) を付与して
    AND で結合する。ユーザー質問内のエンティティを DB の値にマッピングする際に
    使い、多少のスペルミスを許容する。
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()


def structured_retriever(question: str, graph: Neo4jGraph, entity_chain) -> str:
    """質問内で言及されたエンティティの近傍を収集する。"""
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:20})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 1000
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el["output"] for el in response])
    return result


def build_retriever(graph: Neo4jGraph, entity_chain, vector_index: Neo4jVector):
    """構造化データと非構造化データを結合する retriever を返す。"""

    def retriever(question: str) -> str:
        print(f"Search query: {question}")
        structured_data = structured_retriever(question, graph, entity_chain)
        unstructured_data = [
            el.page_content for el in vector_index.similarity_search(question)
        ]
        final_data = f"""Structured data:
        {structured_data}
        Unstructured data:
        {"#Document ".join(unstructured_data)}
        """
        return final_data

    return retriever


def build_chain(llm: ChatOpenAI, retriever):
    """質問に回答する RAG チェーンを組み立てる。"""
    _search_query = RunnableLambda(lambda x: x["question"])

    template = """あなたは優秀なAIです。下記のコンテキストを利用してユーザーの質問に丁寧に答えてください。
必ず文脈からわかる情報のみを使用して回答を生成してください。
{context}

ユーザーの質問: {question}"""
    prompt = ChatPromptTemplate.from_template(template)

    return (
        RunnableParallel(
            {
                "context": _search_query | retriever,
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )


def setup(file_path: str = "ptsd_story.txt"):
    """環境読み込みからチェーン構築までを一括で行う。

    Returns:
        (graph, chain) のタプル。`chain.invoke({"question": ...})` で質問できる。
    """
    load_environment()

    llm = build_llm()
    documents = load_documents(file_path)
    graph = build_graph(documents, llm)

    vector_index = build_vector_index()
    graph.query(
        "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]"
    )

    entity_chain = build_entity_chain(llm)
    retriever = build_retriever(graph, entity_chain, vector_index)
    chain = build_chain(llm, retriever)

    return graph, chain


def ask(chain, question: str) -> str:
    """組み立てたチェーンに質問して回答を得る。"""
    return chain.invoke({"question": question})


if __name__ == "__main__":
    graph, chain = setup("ptsd_story.txt")
    answer = ask(chain, "What treatment helped the author recover from PTSD?")
    print(answer)
