"""Write a subsample of a trial file (the first n unordered pairs, all their trials) for slower guessers.

Usage: python subsample_trials.py <n_pairs> <out_dir> <trials.jsonl> [...]
"""
import json
import sys
from pathlib import Path

n = int(sys.argv[1])
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
for f in map(Path, sys.argv[3:]):
    trials = [json.loads(l) for l in open(f)]
    keys = []
    for t in trials:
        k = tuple(sorted([t["target_id"], t["other_id"]]))
        if k not in keys:
            keys.append(k)
    # spread over the file so all words/premises are represented
    step = max(1, len(keys) // n)
    keep = set(keys[::step][:n])
    with open(out / f.name, "w") as o:
        for t in trials:
            if tuple(sorted([t["target_id"], t["other_id"]])) in keep:
                o.write(json.dumps(t) + "\n")
