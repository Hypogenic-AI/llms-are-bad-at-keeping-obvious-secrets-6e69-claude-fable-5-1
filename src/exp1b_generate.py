"""Exp 1 addendum: token-length-matched filler conditions.

The outlines in `plan_self`/`plan_blind` all ran into the 500-token cap, while the first fillers averaged about 300
tokens. This script generates fillers under the same 500-token cap (asking for about 450 words) and writes the
stories for `filler_craft_long` (the unrelated-text variant was dropped for GPU time), appending them to results/exp1/stories.jsonl.
"""
from common import *

N_REP = 8
REQ_CRAFT_LONG = REQ_CRAFT.replace("About 250 words", "About 450 words")
REQ_IRR_LONG = REQ_IRRELEVANT.replace("About 250 words", "About 450 words")


def main():
    set_seed(SEED + 3)
    model, tok = load_model()
    path = RESULTS / "exp1/stories.jsonl"
    rows = [r for r in load_jsonl(path) if not r["cond"].endswith("_long")]
    cells = [(w, r) for w in WORDS for r in range(N_REP)]
    n = len(cells)
    stage1 = {
        "filler_craft_long": [chat(tok, msgs(SYS_NO_SECRET, REQ_CRAFT_LONG)) for _ in range(n)],
    }
    turn1 = {k: generate(model, tok, v, max_new_tokens=500, desc=f"turn1 {k}") for k, v in stage1.items()}
    rng = random.Random(SEED + 3)
    for k in turn1:
        rng.shuffle(turn1[k])
    new = []
    for i, (w, r) in enumerate(cells):
        s = SYS_DONT_REVEAL.replace("X", w)
        conv = {
            "filler_craft_long": msgs(s, REQ_CRAFT_LONG, turn1["filler_craft_long"][i], TASK_AFTER_CRAFT),
        }
        for c, m in conv.items():
            new.append({"id": f"{c}|{w}|{r}", "cond": c, "word": w, "rep": r, "turn1": turn1[c][i],
                        "messages": m, "prompt": chat(tok, m)})
    stories = generate(model, tok, [r["prompt"] for r in new], max_new_tokens=900, desc="stories")
    for r, st in zip(new, stories):
        r["story"] = st
    save_jsonl(path, rows + new)
    print("saved", len(new))


if __name__ == "__main__":
    main()
