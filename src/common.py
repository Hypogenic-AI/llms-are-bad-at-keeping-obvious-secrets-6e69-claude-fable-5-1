"""Shared stimuli, prompts, model loading and batched generation for all experiments."""
import json
import os
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")  # no C compiler here for torch's Triton kernels
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MODEL_ID = "google/gemma-3-12b-it"
SEED = 42

_sw = json.load(open(ROOT / "datasets/secret_words/secret_words.json"))
WORDS = sum(_sw["holtzman_west_2026_curated"].values(), [])  # 15 curated words
DECOY = _sw["holtzman_west_2026_decoy_pairs"]
_pr = json.load(open(ROOT / "datasets/secret_words/holtzman_west_prompts.json"))

# Writer prompts (Holtzman & West, verbatim)
SYS_DONT_REVEAL = _pr["writer_system"]["dont_reveal"]
SYS_NO_SECRET = _pr["writer_system"]["no_secret"]
SYS_DECOY = _pr["writer_system"]["decoy"]
TASK_STORY = _pr["task"]["story"]

# Turn-1 requests for the plan and filler conditions (ours)
REQ_OUTLINE = ("Before writing, make a plan. Write a detailed outline for a short story (5–8 paragraphs): "
               "the setting, the characters, and what happens in each paragraph. About 250 words. "
               "Just write the outline, nothing else.")
REQ_BRIEF = ("Before writing, come up with a premise for a short story. State it in one or two sentences. "
             "Just write the premise, nothing else.")
REQ_CRAFT = ("Before writing, give some general advice on prose craft for fiction writers (sentence rhythm, "
             "word choice, pacing, showing versus telling). About 250 words. Do not mention any specific plot, "
             "character, setting, object or topic. Just write the advice, nothing else.")
REQ_IRRELEVANT = "Write a detailed explanation of {topic}. About 250 words. Just write the explanation, nothing else."
TASK_AFTER_PLAN = "Now write the short story (5–8 paragraphs) following your outline. Just write the story, nothing else."
TASK_AFTER_BRIEF = "Now write the short story (5–8 paragraphs) based on your premise. Just write the story, nothing else."
TASK_AFTER_CRAFT = "Now write a short story (5–8 paragraphs). Just write the story, nothing else."
TASK_AFTER_IRRELEVANT = "Now, on to a different task. Write a short story (5–8 paragraphs). Just write the story, nothing else."

FILLER_TOPICS = [
    "how municipal water treatment works", "how a bill becomes law in a parliamentary system",
    "how double-entry bookkeeping works", "how vaccines train the immune system",
    "how a four-stroke engine works", "how plate tectonics shapes continents",
    "how the offside rule works in football", "how bread dough rises",
    "how a search engine ranks pages", "how glaciers form and move",
    "how compound interest accumulates", "how the human eye focuses light",
    "how concrete cures and gains strength", "how elections use proportional representation",
    "how a refrigerator moves heat", "how bees make honey",
    "how tides are produced", "how paper is recycled",
    "how a suspension bridge carries load", "how photosynthesis converts light to sugar",
    "how email is routed between servers", "how cheese is aged",
    "how a thermostat regulates temperature", "how soil forms from rock",
    "how airline pilots navigate", "how the printing press worked",
    "how antibiotics kill bacteria", "how batteries store energy",
    "how wool is spun into yarn", "how weather forecasts are made",
]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model():
    """Load Gemma 3 12B-it in bf16 on the GPU. Returns (model, tokenizer)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return model, tok


def get_layers(model):
    """Return the ModuleList of decoder layers (works for the multimodal and text-only wrappers)."""
    best = None
    for name, mod in model.named_modules():
        if name.endswith("layers") and isinstance(mod, torch.nn.ModuleList) and "vision" not in name:
            if best is None or len(mod) > len(best):
                best = mod
    return best


def chat(tok, messages):
    """Render a list of {'role','content'} messages to a prompt string ending in the model turn."""
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def msgs(system, *turns):
    """Build messages: system prompt then alternating user/assistant turns."""
    out = [{"role": "system", "content": system}]
    for i, t in enumerate(turns):
        out.append({"role": "user" if i % 2 == 0 else "assistant", "content": t})
    return out


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens=800, batch_size=32, temperature=1.0, desc="", on_batch=None, strip=True):
    """Sample one completion per prompt string. Sorted by length to limit padding."""
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outs = [None] * len(prompts)
    b = 0
    while b < len(order):
        # long prompts (two-turn conditions) get smaller batches to stay inside GPU memory
        bs = batch_size if len(prompts[order[b]]) < 1500 else max(8, int(batch_size * 0.7))
        idx = order[b:b + bs]
        b += bs
        if on_batch is not None:
            on_batch(idx)  # lets an intervention select per-example directions for this batch
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                             top_k=64, top_p=0.95)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, t in zip(idx, texts):
            outs[i] = t.strip() if strip else t
        print(f"[gen {desc}] {min(b, len(order))}/{len(order)}", flush=True)
    return outs


def save_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]
