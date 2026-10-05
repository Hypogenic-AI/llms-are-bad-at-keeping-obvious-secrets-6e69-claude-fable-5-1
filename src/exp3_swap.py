"""Exp 3c: when does the secret have to be in context for the story to leak? (context swap)

Removing the secret from the context is a complete, quality-preserving ablation of everything the secret adds to
the residual stream at later positions. Two families, K in {30, 100} tokens, "don't reveal" prompt:
  swap_earlyK  the first K story tokens are written with the secret in context; the rest is written by the same
               model continuing that text under the no-secret context (secret present only early)
  swap_lateK   the first K tokens are written under the no-secret context; the rest with the secret in context
               (secret present only late)
Compare with `abl_none` (secret throughout). Rows are appended to results/exp3/ablation_stories.jsonl.
"""
import sys

from common import *

N_REP = int(sys.argv[1]) if len(sys.argv) > 1 else 6
OUT = RESULTS / "exp3"


def main():
    set_seed(SEED + 6)
    model, tok = load_model()
    cells = [(w, r) for w in WORDS for r in range(N_REP)]
    p_secret = [chat(tok, msgs(SYS_DONT_REVEAL.replace("X", w), TASK_STORY)) for w, _ in cells]
    p_none = [chat(tok, msgs(SYS_NO_SECRET, TASK_STORY)) for _ in cells]
    rows = [r for r in load_jsonl(OUT / "ablation_stories.jsonl") if not r["cond"].startswith("swap_")]
    for K in (30, 100):
        for name, first, second in ((f"swap_early{K}", p_secret, p_none), (f"swap_late{K}", p_none, p_secret)):
            prefix = generate(model, tok, first, max_new_tokens=K, desc=name + " prefix", strip=False)
            prefix = [p.lstrip() for p in prefix]
            rest = generate(model, tok, [s + p for s, p in zip(second, prefix)], max_new_tokens=900 - K,
                            desc=name + " rest", strip=False)
            for (w, r), p, q, pr in zip(cells, prefix, rest, second):
                rows.append({"id": f"{name}|{w}|{r}", "cond": name, "word": w, "rep": r, "prompt": pr,
                             "prefix": p, "story": (p + q).strip()})
            save_jsonl(OUT / "ablation_stories.jsonl", rows)
    print("saved")


if __name__ == "__main__":
    main()
