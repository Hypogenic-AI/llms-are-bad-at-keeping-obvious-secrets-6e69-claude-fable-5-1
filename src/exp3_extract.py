"""Exp 3a: read the residual stream at story positions.

Each Exp 1 story is teacher-forced twice: under its real context (with the secret) and under a
text-matched control context in which only the system prompt is replaced by the no-secret one.
The difference between the two removes whatever the story text itself carries about the secret.

Saved to results/exp3/acts_<cond>.npz:
  mean_s, mean_n : [n_story, n_layer+1, d]  mean residual over story positions (secret / no-secret context)
  dec_s, dec_n   : [n_story, len(DEC_LAYERS), 10, d]  mean residual per story decile
  lp_s, lp_n     : [n_story, len(LENS_LAYERS), 10, 15]  logit-lens log-prob of each candidate word
                   (first token, with leading space) per decile; last lens layer is the real output
  ids            : story ids
"""
import sys

from common import *

DEC_LAYERS = [8, 16, 24, 32, 40, 48]
LENS_LAYERS = [16, 24, 32, 40, 48]
CONDS = ["dont_reveal", "decoy", "plan_self", "plan_blind", "plan_brief", "filler_craft", "filler_irrelevant",
         "filler_craft_long", "filler_irrelevant_long", "filler_irrelevant_plain"]
OUT = RESULTS / "exp3"


@torch.no_grad()
def run(model, tok, prompt, story, word_tok, final_norm, lm_head):
    p = tok.encode(prompt, add_special_tokens=False)
    s = tok.encode(story, add_special_tokens=False)
    ids = torch.tensor([p + s], device="cuda")
    out = model(input_ids=ids, output_hidden_states=True)
    hs = out.hidden_states  # tuple of [1, T, d]; the last entry is after the final norm
    assert torch.allclose(lm_head(hs[-1][0, -1]).float(), out.logits[0, -1].float(), atol=0.5), "last hidden state is not post-norm"
    sl = slice(len(p), len(p) + len(s))
    H = torch.stack([h[0, sl].float() for h in hs])  # [L+1, S, d]
    S = H.shape[1]
    bins = torch.clamp((torch.arange(S, device="cuda") * 10) // S, max=9)
    onehot = torch.nn.functional.one_hot(bins, 10).float()  # [S, 10]
    onehot = onehot / onehot.sum(0, keepdim=True).clamp(min=1)
    mean = H.mean(1)
    dec = torch.einsum("lsd,sb->lbd", H[DEC_LAYERS], onehot)
    lps = []
    for l in LENS_LAYERS:
        logits = lm_head(final_norm(hs[l][0, sl]) if l < len(hs) - 1 else hs[l][0, sl]).float()
        lp = torch.log_softmax(logits, -1)[:, word_tok]  # [S, 15]
        lps.append(onehot.T @ lp)  # [10, 15]
    # float32: one residual dimension exceeds the float16 range from layer 25 on
    return mean.cpu().numpy(), dec.cpu().numpy(), torch.stack(lps).cpu().numpy()


def main():
    conds = sys.argv[1:] or CONDS
    model, tok = load_model()
    lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
    final_norm, lm_head = lm.norm, model.lm_head
    word_tok = [tok.encode(" " + w, add_special_tokens=False)[0] for w in WORDS]
    print("word first tokens:", [tok.decode([t]) for t in word_tok])
    rows = load_jsonl(RESULTS / "exp1/stories.jsonl")
    for c in conds:
        sub = [r for r in rows if r["cond"] == c]
        res = {k: [] for k in ["mean_s", "dec_s", "lp_s", "mean_n", "dec_n", "lp_n"]}
        for i, r in enumerate(sub):
            ctrl = [{"role": "system", "content": SYS_NO_SECRET}] + r["messages"][1:]
            for tag, prompt in (("s", r["prompt"]), ("n", chat(tok, ctrl))):
                m, d, lp = run(model, tok, prompt, r["story"], word_tok, final_norm, lm_head)
                res["mean_" + tag].append(m); res["dec_" + tag].append(d); res["lp_" + tag].append(lp)
            if i % 20 == 0:
                print(c, i, len(sub), flush=True)
        np.savez(OUT / f"acts_{c}.npz", ids=np.array([r["id"] for r in sub]),
                 **{k: np.stack(v) for k, v in res.items()})


if __name__ == "__main__":
    (RESULTS / "exp3").mkdir(parents=True, exist_ok=True)
    main()
