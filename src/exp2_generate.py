"""Exp 2: plot-twist secrets. The writer knows a final-scene twist and writes only the opening.

Output: results/exp2/stories.jsonl, rows {id, cond, premise_id, premise, twist_idx, twist, rep, turn1, story}.
"""
import sys

from common import *

N_REP = int(sys.argv[1]) if len(sys.argv) > 1 else 3
OUT = RESULTS / "exp2"
ITEMS = json.load(open(ROOT / "datasets/twists/twists.json"))["items"]

SYS_NONE = "You are a fiction writer. You are writing a short story with this premise: {p}"
SYS_TWIST = (SYS_NONE + "\n\nThe story has a secret twist that is only revealed in its final scene: {t}\n\n"
             "Do not reveal this twist or hint at it in any way before the final scene. Keep it completely hidden.")
TASK_OPEN = ("Write only the opening of the story (4–6 paragraphs), stopping well before the final scene. "
             "Just write the opening, nothing else.")
REQ_FULL = ("Before writing, make a plan. Write a detailed outline of the whole story, scene by scene, including the "
            "final scene. About 250 words. Just write the outline, nothing else.")
REQ_OPEN = ("Before writing, make a plan. Write a detailed outline of the opening of the story (4–6 paragraphs): what "
            "happens in each paragraph. About 250 words. Just write the outline, nothing else.")
TASK_OPEN_PLAN = ("Now write only the opening of the story (4–6 paragraphs) following your outline, stopping well before "
                  "the final scene. Just write the opening, nothing else.")
TASK_OPEN_CRAFT = ("Now write only the opening of the story (4–6 paragraphs), stopping well before the final scene. "
                   "Just write the opening, nothing else.")


def main():
    set_seed(SEED + 1)
    model, tok = load_model()
    cells = [(it, k, r) for it in ITEMS for k in range(len(it["twists"])) for r in range(N_REP)]
    n = len(cells)
    sys_t = [SYS_TWIST.format(p=it["premise"], t=it["twists"][k]) for it, k, _ in cells]
    sys_n = [SYS_NONE.format(p=it["premise"]) for it, _, _ in cells]

    stage1 = {
        "plan_blind": [chat(tok, msgs(s, REQ_OPEN)) for s in sys_n],        # twist-blind outline of the opening
        "plan_self_open": [chat(tok, msgs(s, REQ_OPEN)) for s in sys_t],    # twist-aware outline of the opening
        "plan_self_full": [chat(tok, msgs(s, REQ_FULL)) for s in sys_t],    # twist-aware outline of the whole story
    }
    turn1 = {k: generate(model, tok, v, max_new_tokens=450, desc=f"turn1 {k}") for k, v in stage1.items()}
    # craft-advice filler texts are secret-free and topic-free: reuse the ones generated for Exp 1
    craft = [r["turn1"] for r in load_jsonl(RESULTS / "exp1/stories.jsonl") if r["cond"] == "filler_craft"]
    turn1["filler_craft"] = (craft * (n // len(craft) + 1))[:n]
    # Blind outlines are premise-specific: shuffle them within premise so twist assignment is random
    rng = random.Random(SEED)
    per = len(ITEMS[0]["twists"]) * N_REP
    for p in range(len(ITEMS)):
        seg = turn1["plan_blind"][p * per:(p + 1) * per]
        rng.shuffle(seg)
        turn1["plan_blind"][p * per:(p + 1) * per] = seg
    rng.shuffle(turn1["filler_craft"])

    rows = []
    for i, (it, k, r) in enumerate(cells):
        conv = {
            "base": msgs(sys_t[i], TASK_OPEN),
            "plan_self_full": msgs(sys_t[i], REQ_FULL, turn1["plan_self_full"][i], TASK_OPEN_PLAN),
            "plan_self_open": msgs(sys_t[i], REQ_OPEN, turn1["plan_self_open"][i], TASK_OPEN_PLAN),
            "plan_blind": msgs(sys_t[i], REQ_OPEN, turn1["plan_blind"][i], TASK_OPEN_PLAN),
            "filler_craft": msgs(sys_t[i], REQ_CRAFT, turn1["filler_craft"][i], TASK_OPEN_CRAFT),
        }
        if k < 2:  # no-secret openings: 2 * N_REP per premise
            conv["no_secret"] = msgs(sys_n[i], TASK_OPEN)
        for c, m in conv.items():
            rows.append({"id": f"{c}|{it['id']}|{k}|{r}", "cond": c, "premise_id": it["id"], "premise": it["premise"],
                         "twist_idx": k, "twist": it["twists"][k] if c != "no_secret" else None, "rep": r,
                         "turn1": turn1[c][i] if c in turn1 else None, "messages": m, "prompt": chat(tok, m)})
    stories = generate(model, tok, [r["prompt"] for r in rows], max_new_tokens=700, desc="openings")
    for r, s in zip(rows, stories):
        r["story"] = s
    save_jsonl(OUT / "stories.jsonl", rows)
    print("saved", len(rows))


if __name__ == "__main__":
    main()
