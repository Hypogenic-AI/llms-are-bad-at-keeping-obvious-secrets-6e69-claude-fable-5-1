"""Local 2AFC guesser: reads next-token logits for '1' vs '2' from a local chat model.

Used because the OpenRouter account ran out of credits (HTTP 402) on the day of the run.
Usage: python local_judge.py <guesser> <trials.jsonl> [<trials.jsonl> ...]
  guesser: g27 (google/gemma-3-27b-it, 4-bit NF4) or g12 (google/gemma-3-12b-it, bf16)
Each trials file has rows with 'system' and 'user'. Output is written next to the input as
<name>.<guesser>.jsonl with added fields 'ans' ('1'/'2') and 'margin' (logit('1') - logit('2')).
The answer is greedy among the two options, i.e. the temperature-0 answer restricted to {1, 2}.
"""
import os
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GUESSER_IDS = {"g27": "google/gemma-3-27b-it", "g12": "google/gemma-3-12b-it"}


def load_guesser(name):
    mid = GUESSER_IDS[name]
    tok = AutoTokenizer.from_pretrained(mid)
    tok.padding_side = "left"
    if name == "g27":
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(mid, quantization_config=q, device_map="cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return model, tok


@torch.no_grad()
def judge(model, tok, trials, batch_tokens=28000, opts=("1", "2")):
    """Fill 'ans' and 'margin' for each trial from the next-token logits of the answer options."""
    ids = [tok.encode(o, add_special_tokens=False)[0] for o in opts]
    prompts = [tok.apply_chat_template([{"role": "system", "content": t["system"]}, {"role": "user", "content": t["user"]}],
                                       tokenize=False, add_generation_prompt=True) for t in trials]
    lens = [len(tok.encode(p, add_special_tokens=False)) for p in prompts]
    order = sorted(range(len(trials)), key=lambda i: -lens[i])
    i = 0
    while i < len(order):
        bs = max(1, batch_tokens // lens[order[i]])
        idx = order[i:i + bs]
        enc = tok([prompts[j] for j in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        # run the backbone only and unembed the last position (full-sequence logits would not fit in memory)
        last = model.model(**enc).last_hidden_state[:, -1]
        logits = model.lm_head(last).float()
        for j, row in zip(idx, logits):
            m = (row[ids[0]] - row[ids[1]]).item()
            trials[j]["margin"] = m
            trials[j]["ans"] = opts[0] if m > 0 else opts[1]
        i += bs
        if (i // bs) % 10 == 0:
            print(f"  judged {min(i, len(order))}/{len(order)}", flush=True)
    return trials


def main():
    name = sys.argv[1]
    files = [Path(f) for f in sys.argv[2:]]
    todo = [f for f in files if not f.with_suffix(f".{name}.jsonl").exists()]
    if not todo:
        return
    model, tok = load_guesser(name)
    for f in todo:
        trials = [json.loads(l) for l in open(f)]
        print(f"{f.name}: {len(trials)} trials", flush=True)
        judge(model, tok, trials)
        slim = [{k: v for k, v in t.items() if k not in ("system", "user")} for t in trials]
        with open(f.with_suffix(f".{name}.jsonl"), "w") as out:
            for t in slim:
                out.write(json.dumps(t) + "\n")
        acc = sum(t["ans"] == t["correct"] for t in trials) / len(trials)
        print(f"{f.name}: accuracy {acc:.3f}", flush=True)


if __name__ == "__main__":
    main()
