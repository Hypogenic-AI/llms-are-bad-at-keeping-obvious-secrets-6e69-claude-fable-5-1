"""Exp 1: generate word-secret stories under plan, filler, decoy and anchor conditions.

Output: results/exp1/stories.jsonl with one row per story:
  {id, cond, word, rep, turn1 (assistant turn-1 text or None), story, prompt}
"""
import sys

from common import *

N_REP = int(sys.argv[1]) if len(sys.argv) > 1 else 8
OUT = RESULTS / "exp1"


def main():
    set_seed()
    model, tok = load_model()
    cells = [(w, r) for w in WORDS for r in range(N_REP)]
    n = len(cells)

    # Stage 1: turn-1 texts. Secret-free ones are transplanted into the secret conversations.
    stage1 = {
        "plan_blind": [chat(tok, msgs(SYS_NO_SECRET, REQ_OUTLINE)) for _ in range(n)],
        "plan_brief": [chat(tok, msgs(SYS_NO_SECRET, REQ_BRIEF)) for _ in range(n)],
        "filler_craft": [chat(tok, msgs(SYS_NO_SECRET, REQ_CRAFT)) for _ in range(n)],
        "filler_irrelevant": [chat(tok, msgs(SYS_NO_SECRET, REQ_IRRELEVANT.format(topic=FILLER_TOPICS[i % len(FILLER_TOPICS)])))
                              for i in range(n)],
        "plan_self": [chat(tok, msgs(SYS_DONT_REVEAL.replace("X", w), REQ_OUTLINE)) for w, _ in cells],
    }
    turn1 = {k: generate(model, tok, v, max_new_tokens=500, desc=f"turn1 {k}") for k, v in stage1.items()}
    # Shuffle the secret-free texts so assignment to words is random
    rng = random.Random(SEED)
    for k in ["plan_blind", "plan_brief", "filler_craft", "filler_irrelevant"]:
        rng.shuffle(turn1[k])

    def req_irr(i):
        return REQ_IRRELEVANT.format(topic="(see text)")

    rows = []
    for i, (w, r) in enumerate(cells):
        s = SYS_DONT_REVEAL.replace("X", w)
        conv = {
            "no_secret": msgs(SYS_NO_SECRET, TASK_STORY),
            "dont_reveal": msgs(s, TASK_STORY),
            "decoy": msgs(SYS_DECOY.replace("X", w).replace("'Y'", f"'{DECOY[w]}'"), TASK_STORY),
            "plan_self": msgs(s, REQ_OUTLINE, turn1["plan_self"][i], TASK_AFTER_PLAN),
            "plan_blind": msgs(s, REQ_OUTLINE, turn1["plan_blind"][i], TASK_AFTER_PLAN),
            "plan_brief": msgs(s, REQ_BRIEF, turn1["plan_brief"][i], TASK_AFTER_BRIEF),
            "filler_craft": msgs(s, REQ_CRAFT, turn1["filler_craft"][i], TASK_AFTER_CRAFT),
            # the generic request text is shown; the topic is evident from the transplanted answer
            "filler_irrelevant": msgs(s, "Write a detailed explanation of a topic of your choice. About 250 words. "
                                         "Just write the explanation, nothing else.",
                                      turn1["filler_irrelevant"][i], TASK_AFTER_IRRELEVANT),
        }
        for c, m in conv.items():
            rows.append({"id": f"{c}|{w}|{r}", "cond": c, "word": w, "rep": r,
                         "turn1": turn1.get(c, [None] * n)[i] if c in turn1 else None,
                         "messages": m, "prompt": chat(tok, m)})
    stories = generate(model, tok, [r["prompt"] for r in rows], max_new_tokens=900, desc="stories")
    for r, s in zip(rows, stories):
        r["story"] = s
    save_jsonl(OUT / "stories.jsonl", rows)
    print("saved", len(rows))


if __name__ == "__main__":
    main()
