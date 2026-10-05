# Literature Review: LLMs are bad at keeping obvious secrets

## Research area overview

The hypothesis is that a model writing a story while holding a secret leaks that secret thematically, that the cause is entanglement between "knowing what is coming" and producing prose, and that an explicit plan to condition on might relieve it. Four bodies of work bear on this:

1. **Behavioral leakage of in-context secrets** (Holtzman & West; Gonen et al.; ConfAIde; White Bear; Chameleon).
2. **Hidden state versus externalized state** (Hangman; Luo et al.).
3. **Planning inside hidden states** (Dong et al.; Wu et al.; Future Lens; Pochinkov et al.).
4. **White-box readout and intervention** (Taboo papers; refusal direction; activation addition; LEACE; Gemma Scope), plus **plan-then-write story generation** (Xie & Riedl; Plan-and-Write; Re3; DOC).

No paper found combines them: nobody has tested a plan condition on secret leakage, and nobody has measured or ablated the secret concept inside a model during story writing.

The ten papers named in the specification were read in full; detailed notes are in `papers/notes/`. The 19 additional papers were screened by abstract only, and statements about them below are at that level.

## Key papers

### 1. Holtzman & West (2026), Can You Keep a Secret? (arXiv 2605.10794)
- **Contribution**: the writer–guesser paradigm we extend. A model gets a secret word and a "do not reveal" instruction, then writes a 5–8 paragraph story.
- **Metrics**: 2AFC discrimination (which of two stories had secret X; chance 50%), 2AFC detection, and free-response guessing in up to 20 rounds.
- **Results**: five of seven frontier models leak at 64–79% discrimination; the literal word never appears. Under "actively hide", every model but DeepSeek falls below chance (the avoidance is itself detectable). Short jokes do not leak. Gemma 3 goes from 59% (4B) to 83% (12B) and 84% (27B); Llama 3.1 8B is at 52%. A secret in the user message leaks more than one in the system prompt. A decoy word partly redirects the leakage.
- **Mechanism offered, not tested**: the model must attend to the secret to avoid it, and a short prompt leaves the secret as the dominant influence on under-determined choices ("entropy budget"). The authors speculate a longer, richer prompt could mask it.
- **Code**: not released ("redacted for review"). Prompts are verbatim in the appendix and transcribed in `datasets/secret_words/holtzman_west_prompts.json`.
- **Relevance**: baseline condition, metrics, word lists and model choice all come from here.

### 2. Cywiński et al. (2025), Taboo model (arXiv 2505.14352)
- **Setup**: 20 fine-tunes of Gemma 2 9B, each hinting at a single-token secret word never present in training data or prompt.
- **Methods**: logit lens at layer 32 (secret-token probability peaks around layers 30–37 during hint generation); Gemma Scope 16k residual SAE at layer 32 with a published word-to-latent table.
- **Results** (Pass@10): logit lens top-5 75%, SAE top-5 55%, naive prompting 0%. A second LLM guessing from the hints alone reaches 95%.
- **Code**: https://github.com/EmilRyd/eliciting-secrets; models under `bcywinski/` on HuggingFace.
- **Relevance**: the readout recipe for "is the secret active in the residual stream".

### 3. Cywiński et al. (2025), Eliciting Secret Knowledge (arXiv 2510.01070)
- **Setup**: Taboo and User Gender on Gemma 2 9B; Secret Side Constraint on Llama 3.3 70B.
- **Methods**: logit lens, activation-to-token-embedding similarity, SAE features ranked by activation × log(1/density); black-box prefill attacks.
- **Results**: for Taboo, adding logit-lens tokens raises auditor success from 42.1% to 84.0%. The same readout on the base model gives no benefit, which is the control to copy. Signal is strongest at the assistant control tokens before generation starts.
- **Code**: https://github.com/cywinski/eliciting-secret-knowledge (includes an in-context-secret evaluation flag).
- **Relevance**: readout code and judge prompts. Neither Taboo paper ablates or steers the secret, so a causal test is new.

### 4. Baldelli et al. (2026), LLMs Can't Play Hangman (arXiv 2601.06973)
- **Claim**: an agent restricted to the public transcript cannot both keep a self-chosen secret unresolved and stay consistent with it.
- **Results**: vanilla agents are 2–12% self-consistent on Hangman; a forced private text memory reaches 76–100%. Letting the model decide when to write memory is clearly worse than forcing it.
- **Caveat**: leakage is measured by string match only, and one model still leaked heavily in a diagnosis task under every method.
- **Relevance**: the strongest evidence that externalized state changes secret-handling behavior, and an argument for supplying the plan instead of letting the model opt in.

### 5. Luo et al. (2026), Lack of Stable Internal Beliefs (arXiv 2603.25187)
- **Setup**: 20-questions game; after each turn a forked context asks which target the model holds.
- **Results**: the probed target changes on 11–100% of turns. Reasoning variants often increase drift.
- **Caveat**: the claim that putting the target in context fixes drift has no experiment behind it; "belief" is read from output log-probabilities, not activations.
- **Relevance**: fork-and-probe protocol; a secret held only implicitly is unstable.

### 6. Mann et al. (2025), Don't Think of the White Bear (arXiv 2511.12381)
- **Setup**: log-probability of a forbidden concept after "Do not mention X" plus distractor text; about 5,000 prompts; nine models.
- **Results**: semantic distractors give the strongest rebound and repetition the weakest. Head ablation in Llama 3 8B points to suppressor heads in early-middle layers and amplifier heads slightly later.
- **Caveat**: rebound is measured against a high-load baseline with no no-instruction control, so recency priming is not ruled out.
- **Relevance**: predicts that a plan semantically close to the secret could worsen leakage.

### 7. Mireshghallah et al. (2024), ConfAIde (arXiv 2310.17884)
- **Setup**: four tiers of contextual-privacy tasks; Tier 3 has 270 scenarios where a secret must be kept from a third party.
- **Results**: with a privacy instruction, Tier 3 worst-case leakage is 0.22 (GPT-4) and 0.93 (ChatGPT). Chain of thought did not help (0.22 → 0.24). String matching misses leaks, so a proxy-model detector is used alongside it.
- **Relevance**: free-form reasoning is not the same as a plan and should be a separate comparison arm.

### 8. Dong et al. (2025), Emergent Response Planning (arXiv 2502.06258)
- **Method**: small MLP probes on prompt-final hidden states predict attributes of the whole future response.
- **Results**: on a TinyStories continuation task, a 4-way probe for which animal will be introduced reaches F1 0.72–0.86 before any output token. Decodability across positions before the keyword is U-shaped.
- **Caveat**: no causal test; greedy decoding only; no code.
- **Relevance**: evidence that story content is planned in hidden state, and a template for a position sweep.

### 9. Wu et al. (2024), Do Language Models Plan Ahead for Future Tokens? (arXiv 2404.00859)
- **Distinction**: pre-caching (computing features now that only help later) versus breadcrumbs (features useful now happen to help later).
- **Results**: clear pre-caching on a synthetic task; on natural language GPT-2 shows a small myopia gap (3.28 vs 3.40 nats), which grows with model size up to 2.8B.
- **Relevance**: frames our two outcomes. If leakage is pre-caching-like, an external plan could relieve it. If it is breadcrumb-like (the secret simply colors present-token features), a plan will not help and only removing the concept will.

### 10. Xie & Riedl (2024), Creating Suspenseful Stories (arXiv 2402.17119)
- **Method**: iterative planning; a ladder from direct prompt to full outline to per-chapter event lists.
- **Results**: suspense win-rate against the direct baseline rises to 84.9% for the full method (human raters). Prompts include withholding instructions ("make sure the protagonist is not aware of the reason until the action is taken").
- **Caveat**: no leakage metric; no code.
- **Relevance**: outline formats and the nearest existing version of the fiction-foreshadowing setting.

### Additional papers (abstract level)
- **Gonen et al. 2025** (2408.06518): semantic leakage, irrelevant prompt content biasing generations, measured in 13 models. The unsuppressed version of our effect.
- **Smilga 2025** (2501.06638): smaller Qwen2.5 models leak less, non-monotonically.
- **Karabag et al. 2025** (2501.19398): in the Chameleon game LLM agents fail to conceal the secret word.
- **Castricato et al. 2024** (2402.07896): "avoid entity X" is hard at inference time; fine-tuning helps.
- **Yang, Li & Holtzman** (2506.17871): alignment tuning concentrates output probability; basis for the entropy-budget argument.
- **Pal et al. 2023** (2311.04897), **Men et al. 2024** (2406.16033), **Pochinkov et al. 2024** (2409.06328): hidden states carry information about upcoming tokens and paragraphs.
- **Arditi et al. 2024** (2406.11717), **Turner et al. 2023** (2308.10248), **Panickssery et al. 2023** (2312.06681), **Belrose et al. 2023** (2306.03819): difference-of-means directions, directional ablation, activation addition, linear erasure.
- **Lieberum et al. 2024** (2408.05147): Gemma Scope SAEs.
- **Yao et al. 2019**, **Yang et al. 2022/2023**, **Mirowski et al. 2022**: plan-then-write pipelines that improve coherence. None measures leakage of withheld information.
- **Tian et al. 2024** (2407.13248): LLM stories lack suspense and tension relative to human ones.

## Common methodologies
- **Writer–guesser evaluation**: Holtzman & West; Taboo papers (LLM guesser from hints); ConfAIde (proxy detector).
- **Fork-and-probe**: Hangman, Luo et al. Branch the context and ask about the hidden state.
- **Residual-stream readout**: logit lens and SAE latents (Taboo papers); probes (Dong et al.).
- **Direction interventions**: difference-of-means plus projection ablation or addition (Arditi et al., Panickssery et al.).
- **Plan-then-write**: outline first, prose conditioned on outline (Xie & Riedl, DOC, Re3).

## Standard baselines
- **No secret**: writer told only "You are a creative writer."
- **Not suppressed**: "Your word is 'X'." Upper bound, 97–100% for most models.
- **Don't reveal**: the main condition.
- **Actively hide** and **decoy**: existing prompt-level mitigations to compare a plan against.
- For white-box readouts: the same readout on a run without the secret, and (for interventions) a random or unrelated-word direction.

## Evaluation metrics
- **2AFC discrimination accuracy**, both presentation orders and both words as target; two-sided binomial test against 50%. Below-chance values are signal too.
- **Free-response exact-word rate** within 20 guesses.
- **Literal mention rate** (string match); expected to be zero.
- **Readout strength per token position**: secret-token logit-lens probability or rank, SAE latent activation, or projection on a concept direction.
- **Story quality** under intervention: needed to show ablation does not just damage the text. No paper above gives a standard for this; an LLM judge or perplexity under the unmodified model are options.

## Datasets in the literature
- Secret word lists (Holtzman & West curated and COCA; 20 Taboo words).
- TinyStories and ROCStories (Dong et al.); WritingPrompts (story-generation work).
- ConfAIde tiers; White Bear negation prompts; Narrative-Discourse synopses with turning points.

## Gaps and opportunities
1. **No plan condition.** Holtzman & West vary instruction, task, placement and decoy, not planning.
2. **A confound waiting in any plan experiment.** A plan adds length and content. The entropy-budget account predicts less leakage from any rich prompt, so a plan must be compared with a length-matched non-plan prompt, and a plan written with the secret in view must be separated from one written without it.
3. **No internal measurement during story writing.** Taboo readouts concern fine-tuned secrets while the model is hinting on purpose. Whether an in-context secret stays active token by token through an unrelated story is unmeasured.
4. **No causal test.** Nobody has ablated the secret concept and checked whether leakage falls to chance.
5. **Foreshadowing in fiction has no leakage metric.** The 2AFC design could be adapted to plot twists, but nothing validated exists.

## Recommendations for our experiment

### Direction ranking (top 3 kept)

| Rank | Direction | Evidence | Information gain | Feasibility |
|---|---|---|---|---|
| 1 | **Plan conditioning, behavioral.** Leakage under no plan, self-written plan, supplied secret-blind plan, length-matched filler and free reasoning. | Direct test of the hypothesis; Hangman suggests externalized state helps, ConfAIde and White Bear suggest it may not | High | High: API plus local generation, published prompts |
| 2 | **Secret activity over generation.** Per-token readout of the secret concept in Gemma 3 12B, with and without a plan. | Taboo readouts work at one layer; Dong et al. give the position-sweep template | High | Medium: model and SAEs are public; layer and readout must be tuned |
| 3 | **Causal ablation.** Remove a secret-concept direction during generation and re-measure leakage and quality. | Arditi et al. show single-direction ablation can remove a behavior; untested for concepts like these | High if it works; informative if it fails | Medium: one direction per word; needs quality control |

Pruned:
- **Plot-twist foreshadowing in fiction**: closest to the motivating example, but there is no validated metric or dataset. Keep as an optional small generalization of direction 1 if time allows.
- **Taboo fine-tuned model organisms**: the secret is in the weights and the model hints deliberately; different question.
- **Private-memory agent scaffolds (Hangman)**: about consistency over turns, not thematic leakage.
- **Myopic training**: needs training from scratch; infeasible and indirect.
- **Attention-head circuit tracing of the leak**: expensive and partly answered by direction 3.
- **Scaling ladder across API models**: already done by Holtzman & West.

### Concrete suggestions
- **Model**: Gemma 3 12B-it is the smallest model with strong reported leakage (83%), its weights are cached locally, and Gemma Scope 2 SAEs exist. Gemma 2 9B has better-documented readouts (layer 32, known latents) but its leakage level is unreported and may be low, given Llama 3.1 8B at 52%. First step should be reproducing the 83% locally.
- **Guesser**: a strong API model through OpenRouter, held fixed across conditions; cross-model reading works (Opus reads DeepSeek at 87%).
- **Samples**: Holtzman & West appear to use one story per word per condition. Several stories per cell would allow confidence intervals per condition difference.
- **Controls for direction 1**: length-matched filler; plan written blind to the secret; secret placed before versus after the plan; a check of how much the plan text itself leaks.
- **Controls for direction 3**: random direction of equal norm; another word's direction; quality check on ablated stories.
- **Interpretation**: report below-chance 2AFC as leakage. A drop to 50% is the only clean success.
