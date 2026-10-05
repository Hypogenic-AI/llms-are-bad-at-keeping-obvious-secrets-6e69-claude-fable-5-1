"""Build 2AFC trial files from generated stories.

Usage: python build_trials.py exp1|exp2|exp3
Writes results/<exp>/trials/<name>.jsonl; each row has system, user, target_id, other_id, order, correct.
"""
import random
import sys

from common import RESULTS, SEED, load_jsonl, save_jsonl
from judge import (SYS_TWIST, SYS_WORD, USER_TWIST_DET, USER_TWIST_DISC, USER_WORD_DET, USER_WORD_DISC,
                   det_trials, disc_trials, make_pairs)

N_PARTNERS = 2  # partners drawn per story; gives about 2 x n_story unique pairs, 4 trials each
DET_CONDS = {"dont_reveal", "decoy", "plan_self", "plan_blind", "filler_craft", "filler_craft_long"}


def word_trials(rows, none_rows, out_dir, conds, field="story", tag="", n_partners=N_PARTNERS):
    for c in conds:
        items = [{"id": r["id"], "secret": r["word"], "text": r[field]} for r in rows if r["cond"] == c]
        if not items:
            continue
        rng = random.Random(SEED)
        pairs = make_pairs(items, n_partners, rng)
        save_jsonl(out_dir / f"disc_{c}{tag}.jsonl", disc_trials(pairs, SYS_WORD, USER_WORD_DISC))
        if none_rows and field == "story" and c in DET_CONDS:
            nn = [{"id": r["id"], "secret": None, "text": r["story"]} for r in none_rows]
            dp = [(it, rng.choice(nn)) for it in items]
            save_jsonl(out_dir / f"det_{c}{tag}.jsonl", det_trials(dp, SYS_WORD, USER_WORD_DET))


def exp1():
    rows = load_jsonl(RESULTS / "exp1/stories.jsonl")
    none = [r for r in rows if r["cond"] == "no_secret"]
    conds = sorted({r["cond"] for r in rows} - {"no_secret"})
    word_trials(rows, none, RESULTS / "exp1/trials", conds)
    # does the decoy word leak? Label each decoy-condition story with its decoy word instead of its secret,
    # dropping pairs in which the other story's own secret is that decoy word
    from common import DECOY
    items = [{"id": r["id"], "secret": DECOY[r["word"]], "word": r["word"], "text": r["story"]}
             for r in rows if r["cond"] == "decoy"]
    pairs = [(a, b) for a, b in make_pairs(items, N_PARTNERS, random.Random(SEED))
             if a["secret"] != b["word"] and b["secret"] != a["word"]]
    save_jsonl(RESULTS / "exp1/trials/disc_decoy_asdecoy.jsonl", disc_trials(pairs, SYS_WORD, USER_WORD_DISC))
    # leakage in the self-written outlines themselves
    word_trials(rows, None, RESULTS / "exp1/trials", ["plan_self"], field="turn1", tag="_outline")


def exp2():
    rows = load_jsonl(RESULTS / "exp2/stories.jsonl")
    out = RESULTS / "exp2/trials"
    none = [r for r in rows if r["cond"] == "no_secret"]
    for c in sorted({r["cond"] for r in rows} - {"no_secret"}):
        items = [{"id": r["id"], "secret": r["twist"], "text": r["story"], "premise": r["premise"]}
                 for r in rows if r["cond"] == c]
        rng = random.Random(SEED)
        pairs = make_pairs(items, 1 if c.startswith("open_") else 2, rng, group="premise")
        ex = lambda x: {"p": x["premise"]}
        save_jsonl(out / f"disc_{c}.jsonl", disc_trials(pairs, SYS_TWIST, USER_TWIST_DISC, extra=ex))
        if c.startswith("open_"):
            continue
        dp = []
        for it in items:
            cands = [r for r in none if r["premise"] == it["premise"]]
            r = rng.choice(cands)
            dp.append((it, {"id": r["id"], "secret": None, "text": r["story"], "premise": r["premise"]}))
        save_jsonl(out / f"det_{c}.jsonl", det_trials(dp, SYS_TWIST, USER_TWIST_DET, extra=ex))


def exp3():
    rows = load_jsonl(RESULTS / "exp3/ablation_stories.jsonl")
    conds = sorted({r["cond"] for r in rows})
    word_trials(rows, None, RESULTS / "exp3/trials", conds, n_partners=1)  # one partner per story to save guesser time


if __name__ == "__main__":
    {"exp1": exp1, "exp2": exp2, "exp3": exp3}[sys.argv[1]]()
