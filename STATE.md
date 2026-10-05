# Research State

- Current phase: `None`
- Pipeline completed: `True`

## Previous phases

resource_finder (succeeded), experiment_runner (succeeded)

## Current phase context

- Phase: `experiment_runner`
- Status: `completed`
- Started: `2026-10-05T00:55:21.408733Z`
- Next steps:
  - Validate the report and experimental artifacts before finalizing.

## Workspace check

- Root: `/workspaces/llms-are-bad-at-keeping-obvious-secrets-6e69-claude-fable-5-1`
- Directory usable: `True`

## Output validation

- Valid: `True`
- Expected: `REPORT.md`
- Missing: None
- Outside workspace: None

## Agent notes

<!-- NEURICO_AGENT_NOTES_START -->
### resource_finder
<!-- NEURICO_AGENT_NOTES_START:resource_finder -->
**Phase:** resource_finder, complete (2026-10-05). All expected artifacts exist: `papers/` (29 PDFs), `datasets/`, `code/` (12 repos), `literature_review.md`, `resources.md`, `.resource_finder_complete`.

**Key findings**
- Base paradigm is Holtzman & West 2026 (arXiv 2605.10794): secret word + "don't reveal" + story; 2AFC discrimination 64-79% for most frontier models; Gemma 3 12B at 83%, 4B at 59%, Llama 3.1 8B at 52%. Notes: `papers/notes/2605.10794_notes.md`. Verbatim prompts and word lists: `datasets/secret_words/`.
- No paper tests a plan/outline condition, measures the secret inside the model during story writing, or ablates it. All three top directions are new.
- Prior evidence points both ways on plans: forced private memory helps consistency (Hangman, 2601.06973), but chain of thought does not reduce leakage (ConfAIde) and semantically related context strengthens rebound (White Bear).
- The ten specified papers were read in full (`papers/notes/`); the other 19 only by abstract.

**Directions kept (top 3)**
1. Behavioral plan conditioning: leakage with no plan, self-written plan, supplied secret-blind plan, length-matched filler, free reasoning.
2. Per-token readout of the secret concept during generation in Gemma 3 12B, with and without a plan.
3. Causal ablation of a secret-concept direction, re-measuring leakage and story quality.

**Directions pruned**
- Plot-twist foreshadowing in fiction: no validated metric or dataset; optional small extension of direction 1 only.
- Taboo fine-tuned model organisms: secret in weights and model hints on purpose; different question.
- Private-memory agent scaffolds: about cross-turn consistency, not thematic leakage.
- Myopic training: requires training from scratch.
- Attention-head circuit tracing: costly, partly covered by direction 3.
- Scaling ladder over API models: already done by Holtzman & West.

**Environment facts for the next phase**
- `.venv` (uv) with pypdf, requests, arxiv, datasets, huggingface_hub. torch/transformers are not installed yet.
- `OPENAI_API_KEY` is rejected by OpenAI. `OPENROUTER_KEY` (that exact name) works; daily limit 150.
- `google/gemma-3-12b-it` weights are in the HF cache (5 shards). SAEs `google/gemma-scope-2-12b-it` exist but are not downloaded.
- One RTX A6000 48 GB. Shared disk had 212 GB free before the 24 GB model download.

**Next phase: experiment_runner**
1. Install torch, transformers, accelerate; load Gemma 3 12B-it.
2. Reproduce the don't-reveal story leakage (reported 83%) on the 15 curated words with an OpenRouter guesser, several stories per word. Everything else depends on this.
3. Add plan conditions with the length-matched and secret-blind controls.
4. Readout, then ablation.

**Unresolved**
- Nothing was run end to end: no story was generated and no repository code was executed.
- Whether Gemma 3 12B leaks locally as reported is unverified; Gemma 2 9B (better tooling) has no reported leakage figure.
- Which layer and readout work for an in-context secret in Gemma 3 is unknown; layer 32 is documented only for Gemma 2 9B Taboo fine-tunes.
- Holtzman & West appear to use one story per word per condition; sample sizes for per-condition differences need to be chosen.
- The extended word lists are only lightly vetted.
<!-- NEURICO_AGENT_NOTES_END:resource_finder -->

### experiment_runner
<!-- NEURICO_AGENT_NOTES_START:experiment_runner -->
**Phase:** experiment_runner, complete (2026-10-05). Artifacts on disk: `planning.md`, `REPORT.md`, `README.md`, `src/`, `results/`, `figures/`.

**Setup actually used**
- Writer: local `google/gemma-3-12b-it` (bf16, one A6000, about 8.5 h).
- Guesser: OpenRouter paid models returned HTTP 402 (no credits), so the primary guesser is local Gemma 3 12B reading "1"/"2" logits (`src/local_judge.py`, about 24k trials). Cross-checks: free `nvidia/nemotron-3-ultra-550b-a55b:free` (513 trials) and local Gemma 3 27B 4-bit (192 trials per condition).

**Key findings (evidence: `results/tables_g12.md`, `results/summary_g12.json`, `results/extra.json`, `results/exp3/readout.json`)**
- Anchor replicates: 84.1% discrimination (paper 83%).
- Secret-blind outline: 49.7%; blind 2-sentence premise 54.7%; self-written outline 81.9% (the outline itself leaks at 83.8%).
- Token-matched craft filler 75.1%, unrelated filler with neutral request 78.3%; decoy 83.1% (no help). Unrelated filler plus "Now, on to a different task" 55.8% (wording effect).
- Secret decodable from residual stream at story positions at 97-100% in every condition, including the blind-outline one.
- Context swap: secret present for first 30 tokens only 52.2%; secret present only after 30/100 tokens 80.6%/77.5%.
- Mean-shift patching of the secret signature: -5 points vs none, -10 vs other-word control; adding it without a secret gives about 54%.
- Plot twists: same direction, weak with the 12B guesser (60.4 base vs 53.8 blind outline, order-cancelled); external guesser reads base at 74.8%.

**Decisions and deviations** (table in REPORT.md Appendix A)
- Order-cancelled accuracy added because the 12B guesser answers "1" in about 90% of twist trials.
- Projection ablation and clamping gave degenerate text; replaced by mean-shift patching and context swap.
- Post hoc conditions: `filler_craft_long`, `filler_irrelevant_plain`, `open_base`, `open_plan_blind`, `swap_*`.

**Unresolved**
- Residual leakage under a blind outline: the external guesser says 63% (n=49), the 27B guesser 41% (n=192), the 12B guesser 50%. Needs frontier or human guessers on the saved stories.
- Twist results are underpowered for the available guessers.
- Attention from story positions to the secret tokens was not tested; SAEs were not used.

**Next phase (if any):** paper writing can use REPORT.md directly. Activations (`results/exp3/acts_*.npz`, 3.4 GB) are git-ignored and can be regenerated with `src/exp3_extract.py`.
<!-- NEURICO_AGENT_NOTES_END:experiment_runner -->

<!-- NEURICO_AGENT_NOTES_END -->
