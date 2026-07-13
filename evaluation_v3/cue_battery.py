"""ひとことリスト（キューバッテリー）の生成。

エージェントに投げかける会話の「ひとこと」を、トラウマとの関係の濃さ（レベル）で
層化して LLM に生成させ、JSON に保存する。レベルが般化カーブの横軸になる。

横軸を埋め込み類似度にしないのは循環性の回避のため（活性の入口も埋め込み一致なので、
横軸まで埋め込みだとカーブの下降が半分自明になる）。レベルづけは LLM 生成＋
専門家（臨床）監修という独立の物差しにする。

レベル定義:
  1 = トラウマ中核  … 被害場面を直接想起させる状況の話しかけ
  2 = 関連        … 加害・危険を連想しうる日常の手がかり
  3 = 日常あいまい  … 通常は中立だが、文脈次第で引っかかりうる
  4 = 中立        … トラウマと無関係な日常会話
  5 = ポジティブ    … 楽しい・安心の話題（回復方向）

実行: uv run python -m evaluation_v3.cue_battery [1レベルあたりの個数=20]
出力: evaluation_v3/data/cue_battery.json
"""

import json
import os
import sys
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from actr_foa_kozak_v2 import config

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(_HERE, "data", "cue_battery.json")

LEVELS = {
    1: "トラウマ中核：被害場面（性暴力・夜道・尾行・襲撃など）を直接想起させる状況を告げる",
    2: "関連：加害や危険を連想しうる日常の手がかり（見知らぬ男性、物音、刃物、夜、一人になる等）に触れる",
    3: "日常あいまい：通常は中立だが文脈次第で引っかかりうる（来客、チャイム、帰宅の遅れ、外出の誘い等）",
    4: "中立：トラウマと無関係な日常会話（天気、食事、買い物、テレビ等）",
    5: "ポジティブ：楽しい・安心・回復を感じる話題（趣味、運動、ねぎらい等）",
}


class Cue(BaseModel):
    text: str = Field(description="相手からエージェント本人への短い話しかけ（日本語・自然な話し言葉・1〜2文）")


class CueSet(BaseModel):
    cues: List[Cue] = Field(default_factory=list)


_SYSTEM = """あなたは心理学実験の刺激材料を作る研究者です。
PTSD 当事者（性暴力被害の経験者・女性・夫と同居）との日常会話で、周囲の人（夫や友人）が
本人に向けて言いそうな「ひとこと」を作ります。これは反応の般化（どこまで引き金になるか）を
測る材料なので、以下を厳守してください:

- 指定されたレベルの定義に**厳密に**合わせる（レベル間の混入が実験を壊す）。
- 日本語の自然な話し言葉。1〜2文・40文字以内程度。
- 場面・対象・感覚（視覚/聴覚/場所/時間帯）を多様に散らし、言い回しの重複を避ける。
- 質問形・報告形・誘い形など文型も散らす。
- 被害を露悪的・具体的に描写しない（レベル1でも「状況を告げる」まで）。
- 診断や治療の話はしない。
"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human",
     "参考として、本人の体験記の冒頭を示します（引き金の傾向の参考。文言の流用はしない）:\n"
     "----\n{narrative}\n----\n\n"
     "レベル {level} 「{level_def}」に該当する話しかけを、ちょうど {n} 個作ってください。"),
])


def generate(n_per_level: int) -> List[dict]:
    llm = ChatOpenAI(model=config.EXTRACT_MODEL, temperature=0.7)
    chain = _PROMPT | llm.with_structured_output(CueSet)

    with open(config.SOURCE_TEXT) as f:
        narrative = f.read()[:4000]

    battery: List[dict] = []
    for level, level_def in LEVELS.items():
        print(f"  レベル {level} を生成中... （{level_def[:16]}…）")
        result = chain.invoke({
            "narrative": narrative, "level": level,
            "level_def": level_def, "n": n_per_level,
        })
        for c in (result.cues if result else [])[:n_per_level]:
            text = c.text.strip()
            if text:
                battery.append({"text": text, "level": level})
    return battery


def load(path: str = DEFAULT_PATH) -> List[dict]:
    with open(path) as f:
        return json.load(f)["cues"]


def main() -> None:
    config.load_env()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    battery = generate(n)
    os.makedirs(os.path.dirname(DEFAULT_PATH), exist_ok=True)
    with open(DEFAULT_PATH, "w") as f:
        json.dump({"levels": LEVELS, "source": config.SOURCE_TEXT, "cues": battery},
                  f, ensure_ascii=False, indent=1)
    counts = {lv: sum(1 for c in battery if c["level"] == lv) for lv in LEVELS}
    print(f"✅ {len(battery)} 個を保存 → {DEFAULT_PATH}  内訳: {counts}")


if __name__ == "__main__":
    main()