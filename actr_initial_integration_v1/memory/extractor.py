"""テキストから構造化エピソードを抽出する。

GraphRAG の「エンティティを自由抽出」する方式と違い、ここでは
「エピソード記憶」という構造を与える:
  1 エピソード = 1 つの出来事。
  各エピソードは context(C-rep) / event / sensory(S-rep) と、
  型付きエンティティ（Person/Place/Emotion/...）への関係を持つ。
"PTSD" のような概念そのものをエンティティにはしない。
"""

from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .. import config

# エンティティ型・関係の語彙（プロンプトに提示して揺れを抑える）
ENTITY_TYPES = ["Person", "Place", "Emotion", "Symptom", "Treatment", "Time", "Object", "Concept"]
RELATIONS = ["INVOLVES", "AT", "FELT", "CAUSED", "ABOUT", "WITH", "DURING"]


class ExtractedEntity(BaseModel):
    name: str = Field(description="エンティティ名（例: 加害者, 電車, 恐怖）")
    type: str = Field(description=f"エンティティの型。次から選ぶ: {ENTITY_TYPES}")
    relation: str = Field(description=f"エピソードからの関係。次から選ぶ: {RELATIONS}")


class ExtractedEpisode(BaseModel):
    context: str = Field(description="C-rep。状況・文脈（どこで・どんな状況か）")
    event: str = Field(description="中核イベント（何が起きたか。1つの出来事）")
    sensory: str = Field(description="S-rep。その時の感覚・感情")
    entities: List[ExtractedEntity] = Field(default_factory=list)


class EpisodeList(BaseModel):
    episodes: List[ExtractedEpisode] = Field(default_factory=list)


_SYSTEM = f"""あなたは、一人称の体験記から「エピソード記憶」を抽出する専門家です。

抽出ルール:
- 1 エピソード = 1 つの具体的な出来事。複数の出来事を 1 つに混ぜない。
- 各エピソードを context(状況) / event(中核イベント) / sensory(感覚・感情) の3要素に分解する。
- 登場する具体物・人物・感情などを型付きエンティティとして列挙する。
  型は {ENTITY_TYPES} から、関係は {RELATIONS} から選ぶ。
- 「PTSD」「回復」などの抽象概念そのものを単独のエンティティにしないこと。
  あくまで具体的な出来事・場所・人物・感覚を抽出する。
- テキストに書かれていないことは創作しない。
"""

_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _SYSTEM), ("human", "次のテキストからエピソードを抽出してください:\n\n{text}")]
)


class EpisodeExtractor:
    def __init__(self, model: str = config.EXTRACT_MODEL):
        llm = ChatOpenAI(model=model, temperature=0.1)
        self._chain = _PROMPT | llm.with_structured_output(EpisodeList)

    def extract(self, text: str) -> List[ExtractedEpisode]:
        result = self._chain.invoke({"text": text})
        return result.episodes if result else []
