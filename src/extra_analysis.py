"""Secondary analyses: other guessers, order-cancelled accuracy, decoy details, pooled strength-leak correlation.

Writes results/extra.json and prints a summary.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, pearsonr

from common import DECOY, RESULTS, WORDS
from stats import accuracy, load, story_scores

out = {}

# 1. Other guessers (external free model 'nemo', local Gemma 3 27B 'g27') next to the 12B guesser on the same trials
other = {}
for exp, sub, tag in [("exp1", "trials", "nemo"), ("exp2", "trials", "nemo"), ("exp1", "trials_g27", "g27"), ("exp2", "trials_g27", "g27")]:
    for f in sorted((RESULTS / exp / sub).glob(f"*.{tag}.jsonl")):
        t = load(f)
        if len(t) < 20:
            continue
        name = f.name.replace(f".{tag}.jsonl", "")
        k = sum(x["ans"] == x["correct"] for x in t)
        ci = binomtest(k, len(t)).proportion_ci(method="wilson")
        row = {"acc": k / len(t), "n": len(t), "wilson_lo": ci.low, "wilson_hi": ci.high,
               "p_vs_50": binomtest(k, len(t), 0.5).pvalue, "pos1_rate": float(np.mean([x["ans"] == "1" for x in t]))}
        g12 = RESULTS / exp / "trials" / f"{name}.g12.jsonl"
        if g12.exists():  # the 12B guesser on exactly the same trials
            ref = {(x["target_id"], x["other_id"], x["order"]): x for x in load(g12)}
            same = [ref[(x["target_id"], x["other_id"], x["order"])] for x in t if (x["target_id"], x["other_id"], x["order"]) in ref]
            if same:
                row["g12_same_trials_acc"] = float(np.mean([x["ans"] == x["correct"] for x in same]))
                row["g12_same_trials_n"] = len(same)
        other[f"{exp}/{name}/{tag}"] = row
out["other_guessers"] = other

# 2. Order-cancelled pair-level accuracy for the 12B guesser (uses logit margins; removes position bias exactly)
oc = {}
for exp in ["exp1", "exp2", "exp3"]:
    for f in sorted((RESULTS / exp / "trials").glob("disc_*.g12.jsonl")):
        by = defaultdict(dict)
        for t in load(f):
            by[(t["target_id"], t["other_id"])][t["order"]] = t["margin"]
        v = [m[0] - m[1] > 0 for m in by.values() if 0 in m and 1 in m]
        oc[f"{exp}/{f.name.replace('.g12.jsonl', '')}"] = {"acc": float(np.mean(v)), "n_pairs_x_targets": len(v),
                                                           "pos1_rate": float(np.mean([t["ans"] == "1" for t in load(f)]))}
out["order_cancelled_g12"] = oc

# 3. Decoy: accuracy for the real secret excluding pairs where the other story's decoy is the target word
f = RESULTS / "exp1/trials/disc_decoy.g12.jsonl"
if f.exists():
    t = [x for x in load(f) if DECOY[x["other_id"].split("|")[1]] != x["target_id"].split("|")[1]]
    out["decoy_real_secret_excl_confusable"] = accuracy(t)
f = RESULTS / "exp1/trials/disc_decoy_asdecoy.g12.jsonl"
if f.exists():
    # ids carry the real secret; cluster on it
    out["decoy_word_as_target"] = accuracy(load(f))

# 4. Pooled within-(condition x word) correlation between internal strength and per-story leakage
import exp3_analyze as E
rows = []
for c in E.CONDS:
    fa, ft = RESULTS / f"exp3/acts_{c}.npz", RESULTS / f"exp1/trials/disc_{c}.g12.jsonl"
    if not (fa.exists() and ft.exists()):
        continue
    a = np.load(fa)
    w, r = E.meta(a["ids"])
    leak = story_scores(load(ft))
    y = np.array([leak[i] for i in a["ids"]])
    ref = np.load(RESULTS / "exp3/acts_dont_reveal.npz")
    rw, _ = E.meta(ref["ids"])
    for l in (24, 32, 40):
        d = a["mean_s"][:, l].astype(np.float32) - a["mean_n"][:, l].astype(np.float32)
        rd = ref["mean_s"][:, l].astype(np.float32) - ref["mean_n"][:, l].astype(np.float32)
        s = E.cv_strength(d, w, r) if c == "dont_reveal" else E.cv_strength(d, w, r, ref=(rd, rw))
        sr = s - np.array([s[w == k].mean() for k in range(15)])[w]
        yr = y - np.array([y[w == k].mean() for k in range(15)])[w]
        rows.append((c, l, sr / (sr.std() + 1e-9), yr / (yr.std() + 1e-9)))
pooled = {}
for l in (24, 32, 40):
    xs = np.concatenate([x for c, ll, x, y in rows if ll == l]); ys = np.concatenate([y for c, ll, x, y in rows if ll == l])
    r_, p_ = pearsonr(xs, ys)
    pooled[l] = {"r": float(r_), "p": float(p_), "n": len(xs)}
out["pooled_strength_leak_within_cond_word"] = pooled

# 5. "Default story" marker: with no secret, Gemma 3 12B nearly always writes about a lighthouse keeper.
#    Share of stories containing "lighthouse" per condition (stories whose secret is 'lighthouse' excluded).
from common import load_jsonl
dflt = {}
for path in [RESULTS / "exp1/stories.jsonl", RESULTS / "exp3/ablation_stories.jsonl"]:
    by = defaultdict(list)
    for r in load_jsonl(path):
        if r.get("word") != "lighthouse":
            by[r["cond"]].append("lighthouse" in r["story"].lower())
    for c, v in by.items():
        dflt[c] = {"lighthouse_rate": float(np.mean(v)), "n": len(v)}
out["default_story_marker"] = dflt

json.dump(out, open(RESULTS / "extra.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k != "order_cancelled_g12"}, indent=1, default=float))
print("default-story marker:", {c: round(v["lighthouse_rate"], 2) for c, v in dflt.items()})
for k, v in oc.items():
    print(f"{k:45s} order-cancelled {v['acc']*100:5.1f} (n={v['n_pairs_x_targets']}), answered '1' in {v['pos1_rate']*100:.0f}% of trials")
