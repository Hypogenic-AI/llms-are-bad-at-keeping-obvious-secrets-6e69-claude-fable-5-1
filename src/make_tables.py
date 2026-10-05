"""Print markdown tables for REPORT.md from results/summary_<guesser>.json and the Exp 3 JSON files."""
import json
import sys

from analysis import LABEL
from common import RESULTS

G = sys.argv[1] if len(sys.argv) > 1 else "g12"
S = json.load(open(RESULTS / f"summary_{G}.json"))


def fmt(v):
    return f"{v['acc']*100:.1f} [{v['lo']*100:.1f}, {v['hi']*100:.1f}]" if v else "n/a"


def cond_table(exp, text_key, extra_cols=()):
    print(f"\n### {exp}\n")
    hdr = ["Condition", "Discrimination % [95% CI]", "Order-cancelled % [95% CI]", "n trials", "'1' rate %",
           "Detection (order-cancelled) %", "Story words"] + [c[0] for c in extra_cols]
    print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
    for c, d in S[exp].items():
        if "disc" not in d:
            continue
        t = S.get(text_key, {}).get(c, {})
        row = [LABEL.get(c, c) + f" (`{c}`)", fmt(d["disc"]), fmt(d["disc_oc"]), str(d["disc"]["n"]),
               f"{d['disc']['pos1_rate']*100:.0f}", fmt(d.get("det_oc")), f"{t.get('words', float('nan')):.0f}"]
        row += [f(c) for _, f in extra_cols]
        print("| " + " | ".join(row) + " |")


def contrast_table(key):
    if key not in S:
        return
    print(f"\n### {key}\n")
    print("| Contrast | Difference (points) [95% CI] | p (bootstrap) | p (Holm) |"); print("|---|---|---|---|")
    for d in S[key]:
        print(f"| {d['contrast']} | {d['diff']*100:+.1f} [{d['lo']*100:+.1f}, {d['hi']*100:+.1f}] | {d['p_boot']:.4f} | {d['p_holm']:.4f} |")


tok = json.load(open(RESULTS / "exp1/token_stats.json")) if (RESULTS / "exp1/token_stats.json").exists() else {}
cond_table("exp1", "exp1_text", [("Turn-1 tokens", lambda c: f"{tok.get(c, {}).get('turn1_tokens', 0):.0f}"),
                                 ("Literal mention %", lambda c: f"{S['exp1_text'].get(c, {}).get('literal', 0)*100:.1f}")])
contrast_table("exp1_contrasts"); contrast_table("exp1_contrasts_oc")
if "exp1_outline_leak" in S:
    print("\noutline leak:", fmt(S["exp1_outline_leak"]))
for k in S:
    if k.startswith("extra_"):
        print(k, fmt(S[k]) if "acc" in S[k] else S[k])
if "exp2" in S:
    cond_table("exp2", "exp2_text")
    contrast_table("exp2_contrasts"); contrast_table("exp2_contrasts_oc"); contrast_table("exp2_contrasts_det")
if "exp3" in S:
    q = json.load(open(RESULTS / "exp3/quality.json")) if (RESULTS / "exp3/quality.json").exists() else {}
    rec = json.load(open(RESULTS / "exp3/recall.json")) if (RESULTS / "exp3/recall.json").exists() else {}
    cond_table("exp3", "exp3_text", [
        ("Rating (1-9)", lambda c: f"{q[c]['rating']['mean']:.2f} ± {q[c]['rating']['sem']:.2f}" if c in q else "n/a"),
        ("NLL/token", lambda c: f"{q[c]['nll']['mean']:.3f} ± {q[c]['nll']['sem']:.3f}" if c in q else "n/a"),
        ("Distinct-2", lambda c: f"{q[c]['distinct2']['mean']:.3f}" if c in q else "n/a"),
        ("Recall /15", lambda c: str(sum(w.lower() in a.lower() for w, a in rec[c].items())) if c in rec else "n/a"),
        ("Literal mention %", lambda c: f"{S['exp3_text'].get(c, {}).get('literal', 0)*100:.1f}")])
    contrast_table("exp3_contrasts"); contrast_table("exp3_contrasts_oc")
