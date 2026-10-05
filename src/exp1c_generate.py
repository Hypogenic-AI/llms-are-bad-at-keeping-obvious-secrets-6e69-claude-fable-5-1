"""Exp 1 addendum 2: unrelated-text filler with the neutral story request.

`filler_irrelevant` asked for the story with "Now, on to a different task. Write a short story ...", while
`filler_craft` used "Now write a short story ...". To separate the filler content from that wording, this condition
(`filler_irrelevant_plain`) reuses the exact filler texts of `filler_irrelevant` with the neutral request.
"""
from common import *


def main():
    set_seed(SEED + 4)
    model, tok = load_model()
    path = RESULTS / "exp1/stories.jsonl"
    rows = [r for r in load_jsonl(path) if r["cond"] != "filler_irrelevant_plain"]
    new = []
    for r in rows:
        if r["cond"] != "filler_irrelevant":
            continue
        m = r["messages"][:-1] + [{"role": "user", "content": TASK_AFTER_CRAFT}]
        new.append({"id": f"filler_irrelevant_plain|{r['word']}|{r['rep']}", "cond": "filler_irrelevant_plain",
                    "word": r["word"], "rep": r["rep"], "turn1": r["turn1"], "messages": m, "prompt": chat(tok, m)})
    stories = generate(model, tok, [r["prompt"] for r in new], max_new_tokens=900, desc="stories")
    for r, st in zip(new, stories):
        r["story"] = st
    save_jsonl(path, rows + new)
    print("saved", len(new))


if __name__ == "__main__":
    main()
