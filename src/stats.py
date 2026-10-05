"""Statistics for 2AFC trial files: accuracy, cluster bootstrap CIs, paired condition differences."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

N_BOOT = 5000


def load(path):
    return [json.loads(l) for l in open(path)]


def order_cancelled(trials):
    """Collapse the two presentation orders of each (target, other) pair into one decision using the logit margins.

    margin = logit('1') - logit('2'). The pair is scored correct when the margin is higher with the target shown
    first than with the target shown second. Any constant preference for answering '1' or '2' cancels exactly.
    Returns pseudo-trials (one per pair and target) that `accuracy` and `diff` accept.
    """
    by = {}
    for t in trials:
        by.setdefault((t["target_id"], t["other_id"]), {})[t["order"]] = t["margin"]
    return [{"target_id": a, "other_id": b, "ans": "1" if m[0] - m[1] > 0 else "2", "correct": "1"}
            for (a, b), m in by.items() if 0 in m and 1 in m]


def cluster_of(sid):
    """Cluster unit: the secret word (Exp 1/3) or the premise (Exp 2); both are field 1 of the id."""
    return sid.split("|")[1]


def prep(trials):
    """Return arrays: correct (0/1), cluster of target, cluster of other."""
    c = np.array([t["ans"] == t["correct"] for t in trials], float)
    a = np.array([cluster_of(t["target_id"]) for t in trials])
    b = np.array([cluster_of(t["other_id"]) for t in trials])
    return c, a, b


def _index(a, b, clusters):
    idx = {c: i for i, c in enumerate(clusters)}
    return np.array([idx[x] for x in a]), np.array([idx[x] for x in b])


def _weights(ia, ib, cnt):
    """Bootstrap weights from resampled cluster counts: a trial's weight is the product of its two clusters'
    counts (when both stories share a cluster, as in Exp 2, the count is used once)."""
    wa, wb = cnt[ia].astype(float), cnt[ib].astype(float)
    return np.where(ia == ib, wa, wa * wb)


def accuracy(trials, seed=0):
    """Accuracy with a 95% cluster-bootstrap CI and a two-sided binomial p-value against 50%."""
    c, a, b = prep(trials)
    clusters = sorted(set(a) | set(b))
    ia, ib = _index(a, b, clusters)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(N_BOOT):
        w = _weights(ia, ib, rng.multinomial(len(clusters), np.ones(len(clusters)) / len(clusters)))
        if w.sum() > 0:
            boots.append((w * c).sum() / w.sum())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"acc": c.mean(), "lo": lo, "hi": hi, "n": len(c),
            "p_binom": binomtest(int(c.sum()), len(c), 0.5).pvalue,
            "pos1_rate": np.mean([t["ans"] == "1" for t in trials])}


def diff(trials_x, trials_y, seed=0):
    """Accuracy(x) - accuracy(y) with a paired cluster bootstrap (same resampled clusters for both)."""
    cx, ax, bx = prep(trials_x)
    cy, ay, by = prep(trials_y)
    clusters = sorted(set(ax) | set(bx) | set(ay) | set(by))
    iax, ibx = _index(ax, bx, clusters)
    iay, iby = _index(ay, by, clusters)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(N_BOOT):
        cnt = rng.multinomial(len(clusters), np.ones(len(clusters)) / len(clusters))
        wx, wy = _weights(iax, ibx, cnt), _weights(iay, iby, cnt)
        if wx.sum() > 0 and wy.sum() > 0:
            boots.append((wx * cx).sum() / wx.sum() - (wy * cy).sum() / wy.sum())
    boots = np.array(boots)
    d = cx.mean() - cy.mean()
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": d, "lo": lo, "hi": hi, "p_boot": max(p, 1 / N_BOOT)}


def story_scores(trials):
    """Per-story leakage: for each target story, the mean order-cancelled margin over its pairs.

    For a (target, other) pair the guesser saw both orders; margin = logit('1') - logit('2').
    score = (margin when target is first - margin when target is second) / 2, positive when the guesser
    attributes the secret to the right story regardless of position.
    """
    by = {}
    for t in trials:
        by.setdefault((t["target_id"], t["other_id"]), {})[t["order"]] = t["margin"]
    out = {}
    for (tid, _), m in by.items():
        if 0 in m and 1 in m:
            out.setdefault(tid, []).append((m[0] - m[1]) / 2)
    return {k: float(np.mean(v)) for k, v in out.items()}


def holm(pvals):
    """Holm-adjusted p-values, in the input order."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
