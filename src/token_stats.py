"""Token lengths of prompts, turn-1 texts and stories per Exp 1 condition -> results/exp1/token_stats.json."""
import json

import numpy as np
from transformers import AutoTokenizer

from common import MODEL_ID, RESULTS, load_jsonl

tok = AutoTokenizer.from_pretrained(MODEL_ID)
rows = load_jsonl(RESULTS / "exp1/stories.jsonl")
out = {}
n = lambda s: len(tok.encode(s, add_special_tokens=False))
for c in sorted({r["cond"] for r in rows}):
    sub = [r for r in rows if r["cond"] == c]
    t1 = [n(r["turn1"]) for r in sub if r["turn1"]]
    st = [n(r["story"]) for r in sub]
    out[c] = {"prompt_tokens": float(np.mean([n(r["prompt"]) for r in sub])), "turn1_tokens": float(np.mean(t1)) if t1 else 0.0,
              "story_tokens": float(np.mean(st)), "story_hit_cap_frac": float(np.mean([s >= 899 for s in st]))}
    print(c, {k: round(v, 2) for k, v in out[c].items()})
json.dump(out, open(RESULTS / "exp1/token_stats.json", "w"), indent=1)
