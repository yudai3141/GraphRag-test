"""テキストから恐怖構造（Foa & Kozak）を抽出する。

v1 の「エピソード＋型付きエンティティ」抽出と違い、ここでは
「刺激・反応・意味づけがどう結びついているか」を fragment 単位で抽出する。
1 fragment = 1 つの引き金シーン:
  - trigger      : 中心の刺激
  - other_cues   : 同時に存在した刺激（trigger と CO_OCCURS）
  - responses    : trigger が引き起こす反応（EVOKES）
  - meanings     : 刺激/反応が本人にとって持つ意味（MEANS）＋どの中核評価へ収束するか(ROLLS_UP)
  - episode      : このシーンに対応する過去の出来事（あれば。RECALLS）

「危険」などの中核評価は固定4種（config.CORE_CODES）から選ばせ、具体的意味づけは自由記述させる。
"""

from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .. import config


class MeaningItem(BaseModel):
    text: str = Field(description="具体的な意味づけ・破局的解釈（例: 攻撃されている, 心臓発作で死ぬ, 自分は無力）")
    core: str = Field(description=f"この意味が収束する中核評価。次から1つ: {config.CORE_CODES}")
    source: str = Field(
        description="この意味を生む元。刺激由来なら 'trigger'、反応由来ならその反応名そのもの"
    )


class FragmentEpisode(BaseModel):
    context: str = Field(description="状況・文脈（どこで・どんな状況か）")
    event: str = Field(description="中核イベント（何が起きたか。1つの出来事）")
    sensory: str = Field(description="その時の感覚・感情")
    valence: str = Field(description="この記憶の感情価。'negative' / 'neutral' / 'positive' から1つ")


class FearFragment(BaseModel):
    trigger: str = Field(description="中心の刺激（恐怖・不快を引き起こす対象や状況）")
    other_cues: List[str] = Field(default_factory=list, description="同時に存在した他の刺激")
    responses: List[str] = Field(default_factory=list, description="身体・感情・行動の反応")
    meanings: List[MeaningItem] = Field(default_factory=list)
    episode: Optional[FragmentEpisode] = Field(
        default=None, description="このシーンに対応する具体的な過去の出来事（あれば）"
    )


class FragmentList(BaseModel):
    fragments: List[FearFragment] = Field(default_factory=list)


_SYSTEM = f"""あなたは、一人称の体験記から「恐怖構造」を抽出する専門家です（Foa & Kozak の恐怖ネットワーク）。
恐怖記憶は、刺激・反応・意味づけがどう結びついているか、として捉えます。

抽出ルール:
- テキストを「引き金となるシーン」= fragment に分ける。1 fragment = 1 つの引き金。
- 各 fragment について次を埋める:
  - trigger: 中心の刺激（例: 電車, 上司の厳しい指摘, 勉強）。
  - other_cues: 同時に存在した他の刺激（例: 人混み, トンネル, 締切）。
  - responses: その刺激で起きた身体・感情・行動の反応（例: 動悸, 身体が固まる, 逃げたい, 恐怖）。
  - meanings: 刺激や反応が本人にとって何を意味したか。具体的な破局的解釈を text に書き、
    それが収束する中核評価 core を {config.CORE_CODES} から1つ選ぶ。
    source は、その意味が刺激由来なら 'trigger'、特定の反応由来ならその反応名そのもの。
  - episode: そのシーンに対応する具体的な出来事があれば context/event/sensory/valence を埋める。
- 中核評価(core)の意味: DANGER=危険, BAD=悪い/自分はダメ, POWERLESS=無力/逃げられない, UNENDING=終わらない。
- 「PTSD」「回復」などの抽象概念そのものをノードにしないこと。具体的な刺激・反応・意味を書く。
- テキストに書かれていないことは創作しない。回復や中立の場面なら valence を positive/neutral にする。
- **出力の言語は必ず日本語**にする。原文が英語でも、trigger / other_cues / responses /
  meanings.text / episode の context・event・sensory はすべて自然な日本語に翻訳して書く
  （固有名詞も日本語化する）。ただし core のコード（DANGER 等）と source の 'trigger' はそのまま。
"""

_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _SYSTEM), ("human", "次のテキストから恐怖構造を抽出してください:\n\n{text}")]
)


class FearStructureExtractor:
    def __init__(self, model: str = config.EXTRACT_MODEL):
        llm = ChatOpenAI(model=model, temperature=0.1)
        self._chain = _PROMPT | llm.with_structured_output(FragmentList)

    def extract(self, text: str) -> List[FearFragment]:
        result = self._chain.invoke({"text": text})
        if not result:
            return []
        # core が語彙外なら BAD に寄せる（頑健化）
        for frag in result.fragments:
            for m in frag.meanings:
                if m.core not in config.CORE_CODES:
                    m.core = "BAD"
        return result.fragments
