"""テキストから恐怖構造（Foa & Kozak）を抽出する。

v1 の「エピソード＋型付きエンティティ」抽出と違い、ここでは
「刺激・反応・意味づけがどう結びついているか」を fragment 単位で抽出する。
1 fragment = 1 つの引き金シーン:
  - trigger      : 中心の刺激
  - other_cues   : 同時に存在した刺激（trigger と CO_OCCURS）
  - responses    : trigger が引き起こす反応（EVOKES）
  - meanings     : 刺激/反応が本人にとって持つ意味（MEANS）。開語彙の解釈命題。
  - episode      : このシーンに対応する過去の出来事（あれば。RECALLS＋束ね BINDS）

意味は閉じた中核評価（DANGER 等）に丸めず、開語彙のまま保持する（Lang/Foa の意味要素に忠実）。
"""

from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .. import config


class MeaningItem(BaseModel):
    text: str = Field(description="本人にとっての意味・解釈（開語彙。例: 見られている, 殺される, 自分は無力, もう安全な場所はない）")
    source: str = Field(
        description="この意味を生む元。刺激由来なら 'trigger'、反応由来ならその反応名そのもの"
    )


class FragmentEpisode(BaseModel):
    context: str = Field(description="状況・文脈（どこで・どんな状況か）")
    event: str = Field(description="中核イベント（何が起きたか。1つの出来事）")
    sensory: str = Field(description="その時の感覚・感情")
    valence: str = Field(description="この記憶の感情価。'negative' / 'neutral' / 'positive' から1つ")


class FearFragment(BaseModel):
    trigger: str = Field(description="中心の刺激。短く原子的な語にする（例: 見知らぬ男性, 電車, トンネル, 音, 引き出し）。出来事の説明を入れない")
    other_cues: List[str] = Field(default_factory=list, description="同時に存在した他の刺激。同じく短く原子的な語")
    responses: List[str] = Field(default_factory=list, description="身体・感情・行動の反応。短い語（例: 動悸, 身体が固まる, 逃げたい, 恐怖）")
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
  - meanings: 刺激や反応が本人にとって何を意味したか。具体的な解釈を text に**開語彙**で書く
    （例: 見られている, 殺される, 自分は無力, もう安全な場所はない）。少数の中核カテゴリーに丸めない。
    source は、その意味が刺激由来なら 'trigger'、特定の反応由来ならその反応名そのもの。
  - episode: そのシーンに対応する具体的な出来事があれば context/event/sensory/valence を埋める。
- 「PTSD」「回復」などの抽象概念そのものをノードにしないこと。具体的な刺激・反応・意味を書く。
- テキストに書かれていないことは創作しない。回復や中立の場面なら valence を positive/neutral にする。
- **刺激(trigger/other_cues)と反応(responses)は「短く原子的で、他のエピソードでも再利用できる語」にする**。
  出来事まるごとの長い説明にしない（それは episode に書く）。
  悪い例:「駐車場で見知らぬ男性から口笛を吹かれたこと」→ 良い例: trigger=「見知らぬ男性」。
  悪い例:「他人の顔が加害者の顔に見えること」→ trigger=「他人の顔」, meanings=「加害者に見張られている」。
  同じ対象は毎回同じ語に正規化して使い回す（例: いつも「見知らぬ男性」「電車」「音」）。
- meanings.text も短い破局的結論にする（例:「加害者に見張られている」「殺される」「自分は無力」）。
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
        return result.fragments
