# Cloned Repositories

All are shallow clones (`--depth 1`). None was specified by the research brief; they were found through the papers. None has been run here; the notes below come from reading READMEs and file listings.

## Most useful for the experiment

### eliciting-secret-knowledge
- URL: https://github.com/cywinski/eliciting-secret-knowledge
- Purpose: code for "Eliciting Secret Knowledge from Language Models" (2510.01070).
- Key files: `elicitation_methods/logit_lens.py`, `residual_tokens.py`, `sae.py`; `prompts/taboo/`; `taboo/evaluate_internalization_taboo.py` (has an `--in-context` flag for a secret given in the prompt).
- Use: reference implementations for reading a secret word out of residual activations. Written for Gemma 2 9B at layer 32.
- Requirements: `requirements.txt`; README also installs `flash-attn`.

### eliciting-secrets
- URL: https://github.com/EmilRyd/eliciting-secrets
- Purpose: code for the earlier Taboo paper (2505.14352).
- Key files: `evaluate_logit_lens.py`, `evaluate_sae_weighted.py`, `evaluate_residual_similarity.py`, `feature_map.py` (word to SAE latent), `taboo_words.txt`.
- Use: simpler scripts than the repo above; the word-to-latent map for Gemma Scope 16k at layer 32.

### refusal_direction
- URL: https://github.com/andyrdt/refusal_direction
- Purpose: difference-of-means direction extraction and directional ablation (Arditi et al., 2406.11717).
- Key files: `pipeline/submodules/generate_directions.py`, `select_direction.py`, `pipeline/utils/hook_utils.py` (ablation and activation-addition hooks), `pipeline/model_utils/gemma_model.py`.
- Use: template for ablating a "secret concept" direction at every layer and position during generation. Model wrappers cover Gemma 1, Llama 2/3, Qwen 1 and Yi; Gemma 3 would need a new wrapper.

### Hangman
- URL: https://github.com/chandar-lab/Hangman
- Purpose: Private State Interactive Tasks and private-working-memory agents (2601.06973).
- Key files: `run_sct_hangman.py`, `src/hangman/agents`, `src/hangman/prompts`.
- Use: prompt patterns for a forced private scratchpad; fork-and-probe evaluation.

## Supporting

### confaide
- URL: https://github.com/skywalker023/confaide
- Purpose: ConfAIde benchmark (2310.17884). `benchmark/tier_*.txt`, `eval.py`.
- Use: optional generalization from single-word secrets to contextual secrets.

### Dont-Think-of-the-White-Bear
- URL: https://github.com/cesium132dot9/Dont-Think-of-the-White-Bear
- Purpose: ironic-negation experiments (2511.12381). `data/prompts_negation.csv` has 5,000 prompts.

### semantic_leakage_project
- URL: https://github.com/smilni/semantic_leakage_project
- Purpose: color-focused semantic-leakage dataset and Qwen2.5 generations (2501.06638).

### LLMChameleon
- URL: https://github.com/mustafakarabag/LLMChameleon
- Purpose: Chameleon hidden-identity game (2501.19398).

### doc-story-generation
- URL: https://github.com/yangkevin2/doc-story-generation
- Purpose: Detailed Outline Control (2212.10077). Hierarchical outline generation prompts.
- Use: outline formats for the plan conditions. The full pipeline targets older OpenAI completion models and is not needed.

### Narrative-Discourse
- URL: https://github.com/PlusLabNLP/Narrative-Discourse
- Purpose: story arcs and turning points in human versus LLM narratives (2407.13248). `data_release/` has 1,638 synopses.
- Use: source of plots with late turning points if the foreshadowing extension is attempted.

### FutureGPT2-public
- URL: https://github.com/wiwu2390/FutureGPT2-public
- Purpose: myopic training (2404.00859). Background only; training-time method.

### CAA
- URL: https://github.com/nrimsky/CAA
- Purpose: Contrastive Activation Addition for Llama 2 (2312.06681).
- Note: `analysis/` and `results/` were deleted from the local clone to save space (about 190 MB); re-clone to restore.

## Libraries to install rather than clone

- `transformers` + plain PyTorch forward hooks are enough for direction extraction and ablation.
- `sae-lens` or direct `huggingface_hub` loading for Gemma Scope SAEs (`google/gemma-scope-2-12b-it` for Gemma 3 12B; `google/gemma-scope-9b-it-res` for Gemma 2 9B).
- `nnsight` or `transformer-lens` are optional; TransformerLens support for Gemma 3 was not checked.
