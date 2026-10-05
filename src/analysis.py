"""Behavioural analysis for all three experiments: tables, planned contrasts, figures.

Usage: python analysis.py [guesser]   (default g12)
Writes results/summary_<guesser>.json and figures/*.png.
"""
import json
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS, ROOT, WORDS, load_jsonl
from stats import accuracy, diff, holm, load, order_cancelled

G = sys.argv[1] if len(sys.argv) > 1 else "g12"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

LABEL = {"dont_reveal": "no plan (anchor)", "decoy": "decoy word", "plan_self": "self-written outline",
         "plan_blind": "secret-blind outline", "plan_brief": "secret-blind premise (brief)",
         "filler_craft": "filler: craft advice (300 tok)", "filler_irrelevant": "filler: unrelated text (300 tok)",
         "filler_irrelevant_plain": "filler: unrelated text (300 tok), neutral request",
         "filler_craft_long": "filler: craft advice (500 tok)", "filler_irrelevant_long": "filler: unrelated text (500 tok)",
         "base": "no plan (anchor)", "plan_self_full": "self outline, whole story",
         "plan_self_open": "self outline, opening", "open_base": "no hide instruction, no plan",
         "open_plan_blind": "no hide instruction, twist-blind outline", "abl_none": "no intervention", "sub_own": "subtract own word's signature",
         "sub_own_k2": "subtract 2x own signature", "sub_other": "subtract other word's signature",
         "sub_random": "subtract random vector (same norm)", "add_own_k1": "no secret + add signature",
         "add_own_k3": "no secret + add 3x signature",
         "swap_early30": "secret in context for first 30 tokens only", "swap_early100": "secret for first 100 tokens only",
         "swap_late30": "secret only after first 30 tokens", "swap_late100": "secret only after first 100 tokens"}


def table(exp, conds, kinds=("disc", "det")):
    out = {}
    for c in conds:
        out[c] = {}
        for k in kinds:
            f = RESULTS / exp / "trials" / f"{k}_{c}.{G}.jsonl"
            if f.exists():
                t = load(f)
                out[c][k] = accuracy(t)                       # per-trial accuracy (the pre-specified metric)
                out[c][k]["pos1_rate"] = float(np.mean([x["ans"] == "1" for x in t]))
                out[c][k + "_oc"] = accuracy(order_cancelled(t))  # order-cancelled, from logit margins
    return out


def contrasts(exp, pairs, kind="disc", oc=False):
    res = []
    prep = order_cancelled if oc else (lambda t: t)
    for x, y in pairs:
        fx, fy = (RESULTS / exp / "trials" / f"{kind}_{c}.{G}.jsonl" for c in (x, y))
        if fx.exists() and fy.exists():
            d = diff(prep(load(fx)), prep(load(fy)))
            d["contrast"] = f"{x} - {y}"
            res.append(d)
    if res:
        for d, p in zip(res, holm([d["p_boot"] for d in res])):
            d["p_holm"] = float(p)
    return res


def barplot(tab, order, title, fname, chance_label="chance", key="disc_oc"):
    order = [c for c in order if c in tab and key in tab[c]]
    if not order:
        return
    acc = [tab[c][key]["acc"] * 100 for c in order]
    lo = [(tab[c][key]["acc"] - tab[c][key]["lo"]) * 100 for c in order]
    hi = [(tab[c][key]["hi"] - tab[c][key]["acc"]) * 100 for c in order]
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(order) + 1.5))
    y = np.arange(len(order))[::-1]
    ax.barh(y, acc, xerr=[lo, hi], color="#4878a8", capsize=3)
    ax.axvline(50, color="k", ls=":", label=chance_label)
    for yi, a in zip(y, acc):
        ax.text(a + 1, yi + 0.25, f"{a:.0f}%", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels([LABEL.get(c, c) for c in order])
    ax.set_xlabel("2AFC discrimination accuracy (%)" + (", order-cancelled" if key.endswith("_oc") else "")
                  + "; bars: 95% bootstrap CI"); ax.set_xlim(30, 105)
    ax.set_title(title, fontsize=10); ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout(); plt.savefig(FIG / fname, dpi=150); plt.close()


def per_word(exp, conds, fname):
    """Heatmap of accuracy by target word and condition."""
    M, used = [], []
    for c in conds:
        f = RESULTS / exp / "trials" / f"disc_{c}.{G}.jsonl"
        if not f.exists():
            continue
        by = defaultdict(list)
        for t in load(f):
            by[t["target_id"].split("|")[1]].append(t["ans"] == t["correct"])
        M.append([np.mean(by[w]) * 100 for w in WORDS]); used.append(c)
    if not M:
        return {}
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(used) + 1.8))
    im = ax.imshow(M, vmin=0, vmax=100, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(15)); ax.set_xticklabels(WORDS, rotation=45, ha="right")
    ax.set_yticks(range(len(used))); ax.set_yticklabels([LABEL.get(c, c) for c in used])
    for i in range(len(used)):
        for j in range(15):
            ax.text(j, i, f"{M[i][j]:.0f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, label="accuracy when this word is the target (%)")
    plt.tight_layout(); plt.savefig(FIG / fname, dpi=150); plt.close()
    return {c: dict(zip(WORDS, map(float, m))) for c, m in zip(used, M)}


def text_stats(path, secret_field):
    rows = load_jsonl(path)
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        out[r["cond"]]["words"].append(len(r["story"].split()))
        if r.get("turn1"):
            out[r["cond"]]["turn1_words"].append(len(r["turn1"].split()))
        if secret_field == "word" and r.get("word") and r["cond"] != "no_secret":
            out[r["cond"]]["literal"].append(r["word"].lower() in r["story"].lower())
    return {c: {k: float(np.mean(v)) for k, v in d.items()} for c, d in out.items()}


def main():
    S = {}
    e1 = ["dont_reveal", "decoy", "plan_self", "plan_blind", "plan_brief", "filler_craft", "filler_irrelevant",
          "filler_irrelevant_plain", "filler_craft_long", "filler_irrelevant_long"]
    S["exp1"] = table("exp1", e1)
    f = RESULTS / "exp1/trials" / f"disc_plan_self_outline.{G}.jsonl"
    if f.exists():
        S["exp1_outline_leak"] = accuracy(load(f))
    E1_PAIRS = [("plan_blind", "dont_reveal"), ("plan_self", "dont_reveal"),
                                             ("plan_brief", "dont_reveal"), ("filler_craft", "dont_reveal"),
                                             ("filler_irrelevant", "dont_reveal"), ("decoy", "dont_reveal"),
                                             ("plan_blind", "filler_craft"), ("plan_blind", "filler_irrelevant"),
                                             ("filler_irrelevant_plain", "dont_reveal"), ("filler_irrelevant_plain", "filler_irrelevant"),
                                             ("filler_craft_long", "dont_reveal"), ("filler_irrelevant_long", "dont_reveal"),
                                             ("plan_blind", "filler_craft_long"), ("plan_blind", "filler_irrelevant_long"),
                                             ("plan_self", "filler_craft_long"), ("plan_self", "filler_irrelevant_long"),
                                             ("plan_self", "plan_blind")]
    S["exp1_contrasts"] = contrasts("exp1", E1_PAIRS)
    S["exp1_contrasts_oc"] = contrasts("exp1", E1_PAIRS, oc=True)
    S["exp1_text"] = text_stats(RESULTS / "exp1/stories.jsonl", "word")
    S["exp1_per_word"] = per_word("exp1", e1, f"exp1_per_word_{G}.png")
    barplot(S["exp1"], e1, "Exp 1: word secrets, Gemma 3 12B writer", f"exp1_discrimination_{G}.png")

    if (RESULTS / "exp2/stories.jsonl").exists():
        e2 = ["base", "plan_self_full", "plan_self_open", "plan_blind", "filler_craft", "open_base", "open_plan_blind"]
        S["exp2"] = table("exp2", e2)
        E2_PAIRS = [("plan_blind", "base"), ("plan_self_full", "base"), ("plan_self_open", "base"),
                                                 ("filler_craft", "base"), ("plan_blind", "filler_craft"),
                                                 ("plan_self_open", "plan_blind"), ("open_base", "base"),
                                                 ("open_plan_blind", "open_base")]
        S["exp2_contrasts"] = contrasts("exp2", E2_PAIRS)
        S["exp2_contrasts_oc"] = contrasts("exp2", E2_PAIRS, oc=True)
        S["exp2_contrasts_det"] = contrasts("exp2", [("plan_blind", "base"), ("filler_craft", "base"),
                                                     ("plan_blind", "filler_craft"), ("plan_self_full", "base")], kind="det", oc=True)
        S["exp2_text"] = text_stats(RESULTS / "exp2/stories.jsonl", None)
        barplot(S["exp2"], e2, "Exp 2: plot-twist secrets, Gemma 3 12B writer", f"exp2_discrimination_{G}.png")

    if (RESULTS / "exp3/ablation_stories.jsonl").exists():
        e3 = ["abl_none", "sub_own", "sub_own_k2", "sub_other", "sub_random", "add_own_k1", "add_own_k3",
              "swap_early30", "swap_early100", "swap_late30", "swap_late100"]
        rows = load_jsonl(RESULTS / "exp3/ablation_stories.jsonl")
        e3 = [c for c in e3 if any(r["cond"] == c for r in rows)]
        S["exp3"] = table("exp3", e3, kinds=("disc",))
        E3_PAIRS = [("sub_own", "abl_none"), ("sub_own_k2", "abl_none"),
                                                 ("sub_other", "abl_none"), ("sub_random", "abl_none"),
                                                 ("sub_own", "sub_other"), ("sub_own", "sub_random"),
                                                 ("add_own_k3", "add_own_k1"),
                                                 ("swap_early30", "abl_none"), ("swap_early100", "abl_none"),
                                                 ("swap_late30", "abl_none"), ("swap_late100", "abl_none"),
                                                 ("swap_early100", "swap_late100")]
        S["exp3_contrasts"] = contrasts("exp3", E3_PAIRS)
        S["exp3_contrasts_oc"] = contrasts("exp3", E3_PAIRS, oc=True)
        S["exp3_text"] = text_stats(RESULTS / "exp3/ablation_stories.jsonl", "word")
        S["exp3_per_word"] = per_word("exp3", e3, f"exp3_per_word_{G}.png")
        barplot(S["exp3"], e3, "Exp 3: ablating / adding the secret direction", f"exp3_ablation_{G}.png")

    json.dump(S, open(RESULTS / f"summary_{G}.json", "w"), indent=1, default=float)
    for exp in ["exp1", "exp2", "exp3"]:
        if exp in S:
            print(f"== {exp} ({G})")
            for c, d in S[exp].items():
                print(f"  {c:18s}", "  ".join(f"{k} {v['acc']*100:5.1f} [{v['lo']*100:4.1f},{v['hi']*100:4.1f}] n={v['n']}" for k, v in d.items()))
            for d in S.get(exp + "_contrasts_oc", []):
                print(f"    [order-cancelled] {d['contrast']:34s} {d['diff']*100:+6.1f} [{d['lo']*100:+5.1f},{d['hi']*100:+5.1f}] p={d['p_boot']:.4f} holm={d['p_holm']:.4f}")
            for d in S.get(exp + "_contrasts", []):
                print(f"    {d['contrast']:34s} {d['diff']*100:+6.1f} [{d['lo']*100:+5.1f},{d['hi']*100:+5.1f}] p={d['p_boot']:.4f} holm={d['p_holm']:.4f}")
    if "exp1_outline_leak" in S:
        print("outline leak", S["exp1_outline_leak"])


if __name__ == "__main__":
    main()
