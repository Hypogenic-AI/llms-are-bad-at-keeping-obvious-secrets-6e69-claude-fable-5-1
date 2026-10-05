"""Exp 3b quality check: are ablated/steered stories still normal stories?

For each story in results/exp3/ablation_stories.jsonl, with the un-intervened Gemma 3 12B:
  nll     mean per-token negative log-likelihood of the story given its own prompt
  rating  expected 1-9 quality rating (softmax over the digit logits of a rating prompt)
Also word count and distinct-bigram ratio. Output: results/exp3/quality.json (per-condition means and SEMs).
"""
from common import *

RATE = ("Rate the following short story for overall writing quality (coherence, fluency, and whether it reads as a "
        "complete, sensible story) on a scale from 1 (incoherent or broken) to 9 (excellent). "
        "Answer with ONLY the number.\n\nStory:\n{s}")


@torch.no_grad()
def main():
    model, tok = load_model()
    rows = load_jsonl(RESULTS / "exp3/ablation_stories.jsonl")
    digits = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(1, 10)]
    vals = torch.arange(1, 10, device="cuda").float()
    for i, r in enumerate(rows):
        p = tok.encode(r["prompt"], add_special_tokens=False)
        s = tok.encode(r["story"], add_special_tokens=False)
        ids = torch.tensor([p + s], device="cuda")
        h = model.model(input_ids=ids).last_hidden_state[0, len(p) - 1:-1]
        lp = torch.log_softmax(model.lm_head(h).float(), -1)
        r["nll"] = -lp[torch.arange(len(s)), torch.tensor(s, device="cuda")].mean().item()
        q = tok.apply_chat_template([{"role": "user", "content": RATE.format(s=r["story"])}], tokenize=False,
                                    add_generation_prompt=True)
        qi = tok(q, return_tensors="pt", add_special_tokens=False).to("cuda")
        lg = model.lm_head(model.model(**qi).last_hidden_state[0, -1]).float()[digits]
        r["rating"] = (torch.softmax(lg, -1) * vals).sum().item()
        w = r["story"].lower().split()
        r["words"] = len(w)
        r["distinct2"] = len(set(zip(w, w[1:]))) / max(1, len(w) - 1)
        if i % 100 == 0:
            print(i, len(rows), flush=True)
    out = {}
    for c in sorted({r["cond"] for r in rows}):
        sub = [r for r in rows if r["cond"] == c]
        out[c] = {k: {"mean": float(np.mean([r[k] for r in sub])),
                      "sem": float(np.std([r[k] for r in sub]) / np.sqrt(len(sub)))}
                  for k in ["nll", "rating", "words", "distinct2"]}
        print(c, {k: round(v["mean"], 3) for k, v in out[c].items()}, flush=True)
    json.dump(out, open(RESULTS / "exp3/quality.json", "w"), indent=1)
    save_jsonl(RESULTS / "exp3/quality_per_story.jsonl",
               [{k: r[k] for k in ["id", "cond", "word", "nll", "rating", "words", "distinct2"]} for r in rows])


if __name__ == "__main__":
    main()
