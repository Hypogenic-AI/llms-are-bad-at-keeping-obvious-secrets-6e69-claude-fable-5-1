"""Exp 3a analysis: is the secret decodable at story positions, where, and does its strength predict leakage?

Reads results/exp3/acts_<cond>.npz and the judged Exp 1 trials. Writes results/exp3/readout.json and figures.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from common import RESULTS, ROOT, WORDS
from exp3_extract import CONDS, DEC_LAYERS, LENS_LAYERS
from stats import load, story_scores

GUESSER = sys.argv[1] if len(sys.argv) > 1 else "g27"
OUT = RESULTS / "exp3"
FIG = ROOT / "figures"
W = {w: i for i, w in enumerate(WORDS)}


def meta(ids):
    words = np.array([W[i.split("|")[1]] for i in ids])
    reps = np.array([int(i.split("|")[2]) for i in ids])
    return words, reps


def cv_decode(X, words, reps):
    """Leave-one-rep-out nearest-centroid (cosine) 15-way accuracy. X: [n, d]."""
    correct = 0
    for r in np.unique(reps):
        tr, te = reps != r, reps == r
        mu = X[tr].mean(0)
        cents = np.stack([X[tr & (words == k)].mean(0) - mu for k in range(15)])
        cents /= np.linalg.norm(cents, axis=1, keepdims=True)
        Z = X[te] - mu
        Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        correct += ((Z @ cents.T).argmax(1) == words[te]).sum()
    return correct / len(words)


def cv_strength(D, words, reps, ref=None):
    """Projection of each story's delta onto its own word's direction (unit norm), cross-validated.

    Direction for word k = mean delta of k's training stories - mean delta of other words' training stories.
    If `ref` (delta, words) is given, directions come from that set instead (transfer; no CV needed).
    Returns the own-word projection minus the mean projection onto the 14 other words' directions.
    """
    out = np.zeros(len(words))
    folds = [None] if ref is not None else np.unique(reps)
    for r in folds:
        if ref is not None:
            Dtr, wtr, te = ref[0], ref[1], np.ones(len(words), bool)
        else:
            Dtr, wtr, te = D[reps != r], words[reps != r], reps == r
        V = np.stack([Dtr[wtr == k].mean(0) - Dtr[wtr != k].mean(0) for k in range(15)])
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        P = D[te] @ V.T  # [n_te, 15]
        own = P[np.arange(te.sum()), words[te]]
        oth = (P.sum(1) - own) / 14
        out[te] = own - oth
    return out


def main():
    res = {}
    A = {c: np.load(OUT / f"acts_{c}.npz") for c in CONDS if (OUT / f"acts_{c}.npz").exists()}
    z = A["dont_reveal"]
    words, reps = meta(z["ids"])
    ms, mn = z["mean_s"].astype(np.float32), z["mean_n"].astype(np.float32)
    L = ms.shape[1]

    # 1. Decoding by layer (dont_reveal): secret context, text-only control context, and their difference
    dec = {"secret_ctx": [], "text_only_ctx": [], "delta": []}
    for l in range(L):
        dec["secret_ctx"].append(cv_decode(ms[:, l], words, reps))
        dec["text_only_ctx"].append(cv_decode(mn[:, l], words, reps))
        dec["delta"].append(cv_decode(ms[:, l] - mn[:, l], words, reps))
    res["decode_by_layer"] = {k: [float(x) for x in v] for k, v in dec.items()}

    # 2. Decoding by story decile at selected layers
    ds, dn = z["dec_s"].astype(np.float32), z["dec_n"].astype(np.float32)
    res["decode_by_decile"] = {}
    for li, l in enumerate(DEC_LAYERS):
        res["decode_by_decile"][l] = {
            "delta": [float(cv_decode(ds[:, li, b] - dn[:, li, b], words, reps)) for b in range(10)],
            "text_only_ctx": [float(cv_decode(dn[:, li, b], words, reps)) for b in range(10)]}

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, lab in [("secret_ctx", "secret in context"), ("text_only_ctx", "same text, no-secret context"),
                   ("delta", "difference (context effect)")]:
        ax[0].plot(range(L), dec[k], label=lab)
    ax[0].axhline(1 / 15, color="gray", ls=":", label="chance (1/15)")
    ax[0].set_xlabel("layer (residual stream)"); ax[0].set_ylabel("15-way decoding accuracy (held-out stories)")
    ax[0].set_title("Decoding the secret word from story-position activations"); ax[0].legend(fontsize=8)
    for l in DEC_LAYERS:
        ax[1].plot(range(1, 11), res["decode_by_decile"][l]["delta"], marker="o", label=f"layer {l}")
    ax[1].axhline(1 / 15, color="gray", ls=":")
    ax[1].set_xlabel("story decile (position)"); ax[1].set_ylabel("15-way accuracy, context effect")
    ax[1].set_title("Context effect across the story"); ax[1].legend(fontsize=8); ax[1].set_ylim(0, 1.05)
    plt.tight_layout(); plt.savefig(FIG / "exp3_decoding.png", dpi=150); plt.close()

    # 3. Logit lens / output probability of the secret word: secret context vs text-matched control
    lens = {}
    for c, a in A.items():
        w, _ = meta(a["ids"])
        lp_s, lp_n = a["lp_s"], a["lp_n"]  # [n, lens_layer, 10, 15]
        idx = np.arange(len(w))
        own_s, own_n = lp_s[idx, :, :, w], lp_n[idx, :, :, w]  # [n, lens_layer, 10]
        mask = np.ones((len(w), 15), bool); mask[idx, w] = False
        oth = (lp_s - lp_n).transpose(0, 3, 1, 2)[mask].reshape(len(w), 14, len(LENS_LAYERS), 10).mean(1)
        lens[c] = {"own_lift_by_layer": (own_s - own_n).mean((0, 2)).tolist(),
                   "other_lift_by_layer": oth.mean((0, 2)).tolist(),
                   "own_lift_by_decile_final": (own_s - own_n)[:, -1].mean(0).tolist(),
                   "own_logp_secret_final": float(own_s[:, -1].mean()), "own_logp_control_final": float(own_n[:, -1].mean()),
                   "frac_stories_lift_positive_final": float(((own_s - own_n)[:, -1].mean(1) > 0).mean())}
    res["logit_lens"] = lens

    # 4. Internal strength by condition (directions from dont_reveal) at each DEC layer and its link to leakage
    ref_delta = {l: ms[:, l] - mn[:, l] for l in DEC_LAYERS}
    strength, corr = {}, {}
    for c, a in A.items():
        w, r = meta(a["ids"])
        d = a["mean_s"].astype(np.float32) - a["mean_n"].astype(np.float32)
        strength[c] = {}
        tf = OUT.parent / f"exp1/trials/disc_{c}.{GUESSER}.jsonl"
        leak = story_scores(load(tf)) if tf.exists() else None
        for l in DEC_LAYERS:
            s = cv_strength(d[:, l], w, r) if c == "dont_reveal" else cv_strength(d[:, l], w, r, ref=(ref_delta[l], words))
            strength[c][l] = {"mean": float(s.mean()), "sem": float(s.std() / np.sqrt(len(s)))}
            if leak is not None:
                y = np.array([leak[i] for i in a["ids"]])
                # remove word means so the correlation is within word
                sr = s - np.array([s[w == k].mean() for k in range(15)])[w]
                yr = y - np.array([y[w == k].mean() for k in range(15)])[w]
                corr.setdefault(c, {})[l] = {"pearson_within_word": float(pearsonr(sr, yr)[0]), "p": float(pearsonr(sr, yr)[1]),
                                             "spearman_within_word": float(spearmanr(sr, yr)[0]),
                                             "pearson_raw": float(pearsonr(s, y)[0])}
    # 5. Is the secret still decodable inside each condition (own centroids, leave-one-rep-out)?
    res["decode_by_condition"] = {}
    for c, a in A.items():
        w, r = meta(a["ids"])
        d = a["mean_s"].astype(np.float32) - a["mean_n"].astype(np.float32)
        res["decode_by_condition"][c] = {l: float(cv_decode(d[:, l], w, r)) for l in DEC_LAYERS}
        # guesser-free leakage measure: decode the secret from the story text alone (no-secret context)
        mn_c = a["mean_n"].astype(np.float32)
        res.setdefault("text_only_decode_by_condition", {})[c] = {l: float(cv_decode(mn_c[:, l], w, r)) for l in [0, 32, 48]}
        res["decode_by_condition"][c]["delta_norm_rel_L32"] = float(
            np.linalg.norm(d[:, 32], axis=1).mean() / np.linalg.norm(a["mean_n"][:, 32].astype(np.float32), axis=1).mean())
    print("decode by condition (L32, L48)", {c: (round(v[32], 2), round(v[48], 2)) for c, v in res["decode_by_condition"].items()})
    print("text-only decode by condition (L0, L32)", {c: (round(v[0], 2), round(v[32], 2)) for c, v in res["text_only_decode_by_condition"].items()})
    res["strength_by_condition"] = strength
    res["strength_vs_leak"] = corr
    json.dump(res, open(OUT / "readout.json", "w"), indent=1)

    print("decode (delta) at layers", {l: round(dec["delta"][l], 3) for l in DEC_LAYERS})
    print("decode (text only) at layers", {l: round(dec["text_only_ctx"][l], 3) for l in DEC_LAYERS})
    print("logit lens own lift (final) by cond", {c: round(v["own_lift_by_layer"][-1], 3) for c, v in lens.items()})
    print("strength layer 24", {c: round(v[24]["mean"], 2) for c, v in strength.items()})
    print("corr", json.dumps(corr.get("dont_reveal", {}), indent=0)[:600])


if __name__ == "__main__":
    main()
