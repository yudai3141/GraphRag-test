"""健常統制用の長い一人称記述を生成する（トラウマなし・日常の感情の起伏は豊富）。

PTSD 記述と同規模のグラフを得るため、恐怖構造抽出器が拾える程度に「具体的な出来事＋
通常の感情（不安・苛立ち・満足・警戒・喜び）」を含む記述を、生活領域ごとに複数生成して連結する。
暴力・虐待・トラウマは一切含めない（統制の要件）。

実行: uv run python -m caps5_graph_structure_eval_v3.generate_healthy
出力: caps5_graph_structure_eval_v3/data/healthy_narrative_long.txt
"""

import os

from langchain_openai import ChatOpenAI

from actr_foa_kozak_v2 import config

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_HERE, "data", "healthy_narrative_long.txt")

TOPICS = [
    "your typical weekday morning and commute, with concrete small events",
    "your work: projects, deadlines, meetings, a presentation, ordinary stress and small successes",
    "your relationship with your partner: cooking, small arguments and making up, quiet evenings",
    "friendships and family: phone calls, visits, a friend's news, ordinary social worries",
    "your hobbies: a class you take, reading, a small garden, losing track of time",
    "health and the body: a cold that passes, a check-up you were nervous about, exercise",
    "ordinary caution and worry: walking home at night, a late train, a bill, everyday risk you manage calmly",
    "weekends and small pleasures: walks by the river, coffee, a pet, an unremarkable good day",
]

SYSTEM = (
    "You write detailed, concrete first-person autobiographical prose about an ordinary, "
    "healthy adult life. Include vivid everyday events and the full range of NORMAL emotions "
    "(mild worry, frustration, satisfaction, nervousness, caution, joy, boredom). "
    "Absolutely NO trauma, violence, abuse, assault, disaster, or life-threatening events. "
    "Keep it mundane and emotionally realistic, with specific sensory and situational detail."
)


def main():
    config.load_env()
    llm = ChatOpenAI(model=config.CHAT_MODEL, temperature=0.8)
    parts = ["A first-person account of an ordinary, healthy life (control narrative — no trauma).\n"]
    for i, topic in enumerate(TOPICS, 1):
        print(f"  [{i}/{len(TOPICS)}] generating: {topic[:40]}...")
        prompt = (f"Write about 450–600 words of first-person prose about {topic}. "
                  "Concrete events, normal emotions, sensory detail. No trauma or violence.")
        resp = llm.invoke([("system", SYSTEM), ("human", prompt)])
        parts.append(resp.content.strip())
    text = "\n\n".join(parts)
    with open(OUT, "w") as f:
        f.write(text)
    print(f"✅ {len(text)} chars / {len(text.split())} words → {OUT}")


if __name__ == "__main__":
    main()
