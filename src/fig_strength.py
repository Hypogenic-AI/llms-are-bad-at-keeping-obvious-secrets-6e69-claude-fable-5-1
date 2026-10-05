"""Figure: internal secret-signal strength vs behavioural leakage, across Exp 1 conditions and across words.

Needs results/exp3/readout.json, results/exp3/acts_*.npz and results/summary_<guesser>.json.
Writes figures/exp3_strength_vs_leak.png and results/exp3/strength_vs_leak_summary.json.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from analysis import LABEL
from common import RESULTS, ROOT, WORDS

G = sys.argv[1] if len(sys.argv) > 1 else "g12"
LAYER = "32"
R = json.load(open(RESULTS / "exp3/readout.json"))
S = json.load(open(RESULTS / f"summary_{G}.json"))

conds = [c for c in R["strength_by_condition"] if c in S["exp1"] and "disc" in S["exp1"][c]]
ref = R["strength_by_condition"]["dont_reveal"][LAYER]["mean"]
x = np.array([R["strength_by_condition"][c][LAYER]["mean"] / ref for c in conds])
xe = np.array([R["strength_by_condition"][c][LAYER]["sem"] / ref for c in conds])
y = np.array([S["exp1"][c]["disc"]["acc"] * 100 for c in conds])
ylo = y - np.array([S["exp1"][c]["disc"]["lo"] * 100 for c in conds])
yhi = np.array([S["exp1"][c]["disc"]["hi"] * 100 for c in conds]) - y

# word level (anchor condition): per-word strength vs per-word accuracy
z = np.load(RESULTS / "exp3/acts_dont_reveal.npz")
words = np.array([i.split("|")[1] for i in z["ids"]])
d = z["mean_s"][:, int(LAYER)].astype(np.float32) - z["mean_n"][:, int(LAYER)].astype(np.float32)
wx = []
for w in WORDS:  # norm of the word-specific mean shift, relative to the residual norm at that layer
    v = d[words == w].mean(0) - d[words != w].mean(0)
    wx.append(np.linalg.norm(v))
wy = [S["exp1_per_word"]["dont_reveal"][w] for w in WORDS]

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].errorbar(x, y, xerr=xe, yerr=[ylo, yhi], fmt="o", color="#4878a8", capsize=3)
for c, xi, yi in zip(conds, x, y):
    ax[0].annotate(LABEL.get(c, c), (xi, yi), fontsize=7, xytext=(4, 4), textcoords="offset points")
ax[0].axhline(50, color="k", ls=":")
ax[0].set_xlabel(f"secret signal at story positions, layer {LAYER}\n(projection on the word's direction, relative to the anchor)")
ax[0].set_ylabel("2AFC discrimination accuracy (%)")
r_c, p_c = pearsonr(x, y); rs_c, ps_c = spearmanr(x, y)
ax[0].set_title(f"Across conditions (n={len(conds)}): r={r_c:.2f}, Spearman={rs_c:.2f}", fontsize=10)
ax[1].scatter(wx, wy, color="#a85448")
for w, a, b in zip(WORDS, wx, wy):
    ax[1].annotate(w, (a, b), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax[1].axhline(50, color="k", ls=":")
r_w, p_w = pearsonr(wx, wy); rs_w, ps_w = spearmanr(wx, wy)
ax[1].set_xlabel(f"norm of the word-specific shift, layer {LAYER} (anchor condition)")
ax[1].set_ylabel("accuracy when the word is the target (%)")
ax[1].set_title(f"Across words (n=15): r={r_w:.2f} (p={p_w:.3f}), Spearman={rs_w:.2f} (p={ps_w:.3f})", fontsize=10)
plt.tight_layout(); plt.savefig(ROOT / "figures/exp3_strength_vs_leak.png", dpi=150)
out = {"layer": LAYER, "conditions": {c: {"rel_strength": float(a), "acc": float(b)} for c, a, b in zip(conds, x, y)},
       "cond_pearson": [float(r_c), float(p_c)], "cond_spearman": [float(rs_c), float(ps_c)],
       "word_pearson": [float(r_w), float(p_w)], "word_spearman": [float(rs_w), float(ps_w)]}
json.dump(out, open(RESULTS / "exp3/strength_vs_leak_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
