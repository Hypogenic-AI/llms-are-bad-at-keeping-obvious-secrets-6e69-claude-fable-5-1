"""External cross-check of the local guesser with a free OpenRouter model (about 1000 requests per day).

Usage: python ext_judge.py <n_pairs> <trials.jsonl> [...]
Takes the first n_pairs (target, other) unordered pairs of each file (4 trials each for discrimination) and
writes <name>.nemo.jsonl next to it.
"""
import json
import sys
from pathlib import Path

from judge import GUESSERS, run_trials

n_pairs = int(sys.argv[1])
for f in map(Path, sys.argv[2:]):
    out = f.with_suffix(".nemo.jsonl")
    trials = [json.loads(l) for l in open(f)]
    keys = []
    for t in trials:
        k = tuple(sorted([t["target_id"], t["other_id"]]))
        if k not in keys:
            keys.append(k)
    keep = set(keys[:n_pairs])
    sub = [t for t in trials if tuple(sorted([t["target_id"], t["other_id"]])) in keep]
    run_trials(sub, GUESSERS["nemo"], concurrency=4)
    done = [{k: v for k, v in t.items() if k not in ("system", "user")} for t in sub if t["ans"]]
    with open(out, "w") as o:
        for t in done:
            o.write(json.dumps(t) + "\n")
    acc = sum(t["ans"] == t["correct"] for t in done) / max(1, len(done))
    print(f.name, "answered", len(done), "of", len(sub), "acc", round(acc, 3), flush=True)
