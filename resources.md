# Resources Catalog

## Summary

Resources for testing whether an explicit plan reduces secret leakage in LLM story writing, and whether the secret concept is active in, and causally responsible through, the model's internal state.

- Papers: 29 (10 specified and read in full, 19 found by search and screened by abstract)
- Datasets: 4 downloaded, plus 4 that ship inside cloned repositories
- Repositories: 12 cloned
- Model weights: `google/gemma-3-12b-it` downloaded to the HuggingFace cache (see "Models and API access")

## Papers

| Title | Authors | Year | File | Key info |
|---|---|---|---|---|
| Can You Keep a Secret? Involuntary Information Leakage | Holtzman, West | 2026 | papers/2605.10794_*.pdf | Paradigm, prompts, 2AFC metric; Gemma 3 12B leaks at 83% |
| Towards eliciting latent knowledge (Taboo) | Cywiński et al. | 2025 | papers/2505.14352_*.pdf | Logit lens and SAE at layer 32, Gemma 2 9B |
| Eliciting Secret Knowledge | Cywiński et al. | 2025 | papers/2510.01070_*.pdf | Readout methods, base-model control |
| LLMs Can't Play Hangman | Baldelli et al. | 2026 | papers/2601.06973_*.pdf | Forced private memory fixes consistency |
| Lack of Stable Internal Beliefs | Luo et al. | 2026 | papers/2603.25187_*.pdf | Implicit targets drift; fork-and-probe |
| Don't Think of the White Bear | Mann et al. | 2025 | papers/2511.12381_*.pdf | Negation primes the concept |
| ConfAIde | Mireshghallah et al. | 2024 | papers/2310.17884_*.pdf | Chain of thought does not reduce leakage |
| Emergent Response Planning | Dong et al. | 2025 | papers/2502.06258_*.pdf | Probes decode future story content |
| Do LMs Plan Ahead for Future Tokens? | Wu et al. | 2024 | papers/2404.00859_*.pdf | Pre-caching versus breadcrumbs |
| Creating Suspenseful Stories | Xie, Riedl | 2024 | papers/2402.17119_*.pdf | Plan-then-write prompts |
| 19 additional papers | | | papers/ | Semantic leakage, future-token planning, steering and ablation, SAEs, outline-based story generation |

See `papers/README.md` for the full list and `papers/notes/` for per-paper notes on the ten specified papers.

## Datasets

| Name | Source | Size | Use | Location |
|---|---|---|---|---|
| Secret words and prompts | Holtzman & West appendices; Taboo repo; Brysbaert norms | 15 + 15 + 20 words, 71 extended; all prompt templates | Core experimental stimuli | datasets/secret_words/ |
| WritingPrompts | HF `euclaise/writingprompts` | 15,138 test, 15,620 validation | Story premises | datasets/writingprompts/ |
| ROCStories | HF `mintujupally/ROCStories` | 78,528 train stories | Neutral narrative text | datasets/rocstories/ |
| Taboo conversations | HF `bcywinski/taboo-*` | 300 per word (4 words), 150 adversarial | Hinting text for concept directions | datasets/taboo/ |
| ConfAIde | in repo | Tier 3: 270 scenarios | Optional generalization | code/confaide/benchmark/ |
| White Bear prompts | in repo | 5,000 prompts | Optional | code/Dont-Think-of-the-White-Bear/data/ |
| Narrative-Discourse | in repo | 1,638 synopses | Optional foreshadowing extension | code/Narrative-Discourse/data_release/ |
| Semantic leakage (colors) | in repo | not counted | Optional | code/semantic_leakage_project/data/ |

See `datasets/README.md` for download instructions.

## Code repositories

| Name | URL | Purpose | Location |
|---|---|---|---|
| eliciting-secret-knowledge | github.com/cywinski/eliciting-secret-knowledge | Logit lens, residual-token and SAE readouts | code/eliciting-secret-knowledge/ |
| eliciting-secrets | github.com/EmilRyd/eliciting-secrets | Taboo scripts, word-to-latent map | code/eliciting-secrets/ |
| refusal_direction | github.com/andyrdt/refusal_direction | Direction extraction and ablation hooks | code/refusal_direction/ |
| Hangman | github.com/chandar-lab/Hangman | Private-memory agents, prompts | code/Hangman/ |
| confaide | github.com/skywalker023/confaide | Benchmark and eval | code/confaide/ |
| Dont-Think-of-the-White-Bear | github.com/cesium132dot9/Dont-Think-of-the-White-Bear | Negation prompts | code/Dont-Think-of-the-White-Bear/ |
| semantic_leakage_project | github.com/smilni/semantic_leakage_project | Semantic-leakage data | code/semantic_leakage_project/ |
| LLMChameleon | github.com/mustafakarabag/LLMChameleon | Chameleon game | code/LLMChameleon/ |
| doc-story-generation | github.com/yangkevin2/doc-story-generation | Outline formats | code/doc-story-generation/ |
| Narrative-Discourse | github.com/PlusLabNLP/Narrative-Discourse | Turning-point data | code/Narrative-Discourse/ |
| FutureGPT2-public | github.com/wiwu2390/FutureGPT2-public | Myopic training | code/FutureGPT2-public/ |
| CAA | github.com/nrimsky/CAA | Contrastive activation addition | code/CAA/ |

See `code/README.md`. None of these was run.

## Models and API access (checked on 2026-10-05)

- **GPU**: one NVIDIA RTX A6000, 48 GB.
- **`OPENAI_API_KEY` is rejected** by the OpenAI API ("Incorrect API key provided"). Do not plan around it.
- **`OPENROUTER_KEY` works** (note the variable name; it is not `OPENROUTER_API_KEY`). A test call to `google/gemma-3-12b-it` succeeded. The key reported a daily limit of 150 with about 146 remaining.
- OpenRouter lists the models Holtzman & West used: `anthropic/claude-opus-4.6`, `anthropic/claude-sonnet-4.6`, `openai/gpt-5.4`, `meta-llama/llama-4-maverick`, `deepseek/deepseek-v3.2`, and `google/gemma-3-{4b,12b,27b}-it`.
- **HuggingFace**: the token can fetch configs for `google/gemma-2-9b-it`, `google/gemma-3-{4b,12b,27b}-it`, `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-14B-Instruct` and `mistralai/Mistral-Nemo-Instruct-2407`. Only the Gemma 3 12B weights were downloaded; cache folders exist for several others but their contents were not inspected beyond size.
- **SAEs**: `google/gemma-scope-2-12b-it` has residual-stream SAEs at layers 12, 24, 31 and 41 in widths 16k, 65k, 262k and 1M under `resid_post/`, and all layers under `resid_post_all/`. Not downloaded. `google/gemma-scope-9b-it-res` covers Gemma 2 9B.
- **Taboo models**: `bcywinski/gemma-2-9b-it-taboo-<word>` for the 20 words. Not downloaded.
- **Disk**: the shared volume showed 99% used with 212 GB free before the model download.

## Resource gathering notes

### Search strategy
- Paper-finder (fast mode) with seven queries covering secret leakage, foreshadowing and planning, secret elicitation, outline-conditioned story generation, concept ablation, and future-token planning.
- The ten specified arXiv IDs were downloaded directly. Further papers came from paper-finder hits and from the reference list of Holtzman & West.
- Code was found from URLs printed in the PDFs and GitHub search; datasets from HuggingFace search.

### Selection criteria
Kept: work that measures leakage of in-context information, work that reads or edits concepts in the residual stream, work on planning in hidden states, and plan-then-write story systems. Dropped: training-data memorization and credential-leak papers that paper-finder returned.

### Challenges
- The arXiv metadata API returned HTTP 429, so titles and authors were taken from the PDFs' first pages.
- Paper-finder returned few highly relevant hits for the core topic (one paper scored 3: Holtzman & West).
- Holtzman & West's code is not public.

### Gaps and workarounds
- No leakage code to reuse: prompts and protocol were transcribed from the paper's appendix so the pipeline can be rebuilt.
- No foreshadowing benchmark: this direction was deprioritized (see below).
- Gemma 3 has less interpretability tooling than Gemma 2: plain PyTorch hooks are the fallback.

## Recommendations for experiment design

1. **Primary stimuli**: the 15 curated Holtzman & West words first, for comparability; the COCA and extended lists for generalization.
2. **Baselines**: no secret, not suppressed, don't reveal; then actively hide and decoy as prompt-level mitigations to compare a plan against.
3. **Metrics**: 2AFC discrimination in both orders (primary), free-response guess rate, literal mention rate; per-token secret readout; story quality under intervention.
4. **Code to adapt**: `eliciting-secret-knowledge/elicitation_methods/` for readouts; `refusal_direction/pipeline/utils/hook_utils.py` for ablation; `doc-story-generation` and the Xie & Riedl prompts (in `papers/notes/2402.17119_notes.md`) for outline formats.
5. **First check**: reproduce the reported 83% discrimination for Gemma 3 12B locally before building anything on it. The top three directions and the pruned ones are listed in `literature_review.md` and `STATE.md`.

## Experiment-runner addendum (2026-10-05)

- OpenRouter paid models returned HTTP 402 ("Insufficient credits") for every model tried, although the key reported a daily limit with headroom. Only `:free` models worked (about 1,000 requests per day; `nvidia/nemotron-3-ultra-550b-a55b:free` was used).
- `google/gemma-3-27b-it` was downloaded to the HF cache and used in 4-bit as a second local guesser.
- New stimuli written for this study: `datasets/twists/twists.json` (12 premises x 4 twists).
- No C compiler is installed, and torch 2.14 tries to JIT a Triton kernel; `TORCH_DISABLE_NATIVE_JIT=1` (set in `src/common.py`) avoids it.
- Gemma 3 12B residual activations overflow float16 in one dimension from layer 25 on; save them as float32.
