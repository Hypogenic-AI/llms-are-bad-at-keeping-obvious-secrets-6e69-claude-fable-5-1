"""Exp 2 addendum: twist known but NO instruction to hide it (natural foreshadowing).

`open_base`: the writer knows the final-scene twist and writes the opening, with no secrecy instruction.
`open_plan_blind`: same, but the opening follows a twist-blind outline (the outlines of `plan_blind` are reused).
These give a positive control for the twist measurement and test the plan effect on unprompted foreshadowing.
"""
from common import *
from exp2_generate import REQ_OPEN, SYS_NONE, TASK_OPEN, TASK_OPEN_PLAN

SYS_OPEN = SYS_NONE + "\n\nThe story has a twist that is only revealed in its final scene: {t}"


def main():
    set_seed(SEED + 5)
    model, tok = load_model()
    path = RESULTS / "exp2/stories.jsonl"
    rows = [r for r in load_jsonl(path) if not r["cond"].startswith("open_")]
    new = []
    for r in rows:
        if r["cond"] != "plan_blind":
            continue
        s = SYS_OPEN.format(p=r["premise"], t=r["twist"])
        for c, m in {"open_base": msgs(s, TASK_OPEN),
                     "open_plan_blind": msgs(s, REQ_OPEN, r["turn1"], TASK_OPEN_PLAN)}.items():
            new.append({"id": f"{c}|{r['premise_id']}|{r['twist_idx']}|{r['rep']}", "cond": c, "premise_id": r["premise_id"],
                        "premise": r["premise"], "twist_idx": r["twist_idx"], "twist": r["twist"], "rep": r["rep"],
                        "turn1": r["turn1"] if c == "open_plan_blind" else None, "messages": m, "prompt": chat(tok, m)})
    stories = generate(model, tok, [r["prompt"] for r in new], max_new_tokens=700, desc="openings")
    for r, st in zip(new, stories):
        r["story"] = st
    save_jsonl(path, rows + new)
    print("saved", len(new))


if __name__ == "__main__":
    main()
