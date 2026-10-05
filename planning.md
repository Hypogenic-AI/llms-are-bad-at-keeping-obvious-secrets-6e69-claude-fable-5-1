# Planning: does a plan stop secrets leaking into stories, and what carries the leak?

## Motivation & Novelty Assessment

### Why This Research Matters
Models that cannot keep in-context information out of their writing foreshadow plots, leak system-prompt content, and make "hidden state" unreliable for games, fiction and agents. If an explicit plan fixed this, it would be a cheap mitigation; if the leak is a readable, removable internal signal, it could be monitored or edited.

### Gap in Existing Work
Holtzman & West (2026) established the behavioural effect (a secret word leaks thematically into stories; Gemma 3 12B at 83% two-alternative discrimination) using API models only. Nobody has tested (1) whether conditioning on a plan reduces leakage, with controls that separate "plan fixes content" from "plan is just more context", (2) plot-level secrets (a twist the writer knows is coming), or (3) whether the secret is present in the residual stream during writing, whether its strength predicts leakage, and whether removing it removes leakage.

### Our Novel Contribution
A white-box study on Gemma 3 12B-it (the smallest model reported to leak strongly) with three parts: plan conditioning with length-matched controls, plot-twist secrets, and readout plus ablation of a secret-concept direction.

### Experiment Justification
- **Exp 1 (word secrets, plan conditions):** the idea's direct question. The plain "don't reveal" condition is the positive anchor, the decoy is the known-mitigation yardstick, and two filler conditions separate content-fixing from added context.
- **Exp 2 (plot-twist secrets):** tests whether the word result carries over to the case the idea is actually about (foreshadowing), with a no-secret control because twists may be guessable from the premise.
- **Exp 3 (mechanism):** readout of the secret at story-token positions (text-matched against a no-secret context), story-level link between readout strength and leakage, and ablation against random and other-concept directions with a quality check.

## Research Question
When Gemma 3 12B writes a story while holding a secret it was told not to reveal, (a) does an explicit outline reduce leakage beyond what length-matched irrelevant context does, for word secrets and for plot twists, and (b) is the secret linearly present in the residual stream throughout generation, does its strength predict leakage, and does ablating it remove leakage?

## Hypothesis Decomposition
- H1: stories written from an outline leak less than stories written with no outline.
- H1a (content-fixing): a secret-blind outline reduces leakage more than length-matched filler. H1b (dilution): filler alone reduces leakage. H1c: a self-written outline (written knowing the secret) may carry the secret itself and so reduce leakage less than a blind one.
- H2: the same pattern holds for twist secrets.
- H3: the secret word is decodable from residual activations at story positions, beyond what the text itself carries, throughout the story.
- H4: readout strength correlates with per-story leakage.
- H5: projecting out the secret's direction reduces leakage more than random or other-concept directions, without degrading the story.

## Proposed Methodology

### Writer
`google/gemma-3-12b-it`, local, bf16, HF transformers, temperature 1.0 with the model's default top-k 64 and top-p 0.95. Prompts follow Holtzman & West verbatim where they exist.

### Exp 1 conditions (15 curated words x 8 stories = 120 stories each)
All plan and filler conditions share one format: turn 1 is a user request and an assistant text, turn 2 asks for the story. The secret is always in the system prompt ("don't reveal" wording).
1. `no_secret`: no secret (for detection trials).
2. `dont_reveal`: positive anchor.
3. `decoy`: known mitigation (paper's decoy pairs).
4. `plan_self`: the model writes an outline (knowing the secret), then the story.
5. `plan_blind`: an outline written by the same model with no secret is placed in the assistant turn; the story follows it.
6. `plan_brief`: as 5 with a one- or two-sentence premise instead of a full outline (dose of content-fixing).
7. `filler_craft`: turn 1 is length-matched generic prose-craft advice (task-relevant, fixes no content).
8. `filler_irrelevant`: turn 1 is length-matched expository text on an unrelated topic.

Contrast logic: 5 vs 7/8 isolates content-fixing from dilution; 7/8 vs 2 measures dilution; 4 vs 5 measures what a secret-aware plan carries. Outlines from condition 4 are themselves scored for leakage.

### Exp 2 (twist secrets)
12 premises x 4 candidate twists, written for this study. The writer knows the twist "will be revealed in the final scene" and writes only the opening, told not to reveal or hint. Conditions: `no_secret`, `base`, `plan_self` (outline of the whole story including the twist, then the opening), `plan_blind` (twist-blind outline of the opening), `filler_craft`. 4 stories per (premise, twist, condition). Trials pair two openings with the same premise and different twists, so premise-level guessability cancels; detection trials pair a twist opening with a no-secret opening.

### Exp 3 (mechanism, word secrets)
- **Readout:** teacher-force each story under its secret context and under the no-secret context. The difference at story positions removes what the text itself carries. Nearest-centroid 15-way decoding of the secret from that difference (held-out stories), by layer and by position decile. Also logit-lens/output probability of the secret word under both contexts.
- **Strength vs leakage:** per-story projection onto the true word's direction against per-story 2AFC score, within word.
- **Ablation:** per-layer difference-of-means direction for each word (from training stories), projected out of the residual stream at every layer during generation. Variants: generated positions only, all positions. Controls: random unit direction, another word's direction. Quality: judge rating, length, NLL under the clean model. Recall probe: can the ablated model still state its secret?

### Measurement
- 2AFC discrimination (primary): two stories with different secrets, "which has secret X?", both presentation orders and both targets. About 180 pairs x 4 = 720 trials per condition. Detection (secret vs none) as secondary.
- Guessers through OpenRouter at temperature 0: `openai/gpt-5.6-terra` (primary) and `google/gemini-3.8-flash` (second guesser), chosen from the live catalog on 2026-10-05 before seeing any condition differences.
- Literal mention rate of the secret word.

### Statistical Analysis Plan
Accuracy with 95% CIs from a cluster bootstrap over secret words (premises for Exp 2), which is the unit that generalizes; condition differences by paired bootstrap over the same clusters. Binomial test against 50% reported for comparability with the paper. Alpha 0.05; Holm correction across the planned contrasts within each experiment.

## Expected Outcomes
Support for H1a: `plan_blind` well below both fillers. Dilution only: plans and fillers equal and below anchor. No effect: all near anchor. For H5: own-direction ablation near 50% with other-direction and random near anchor.

## Timeline
Setup and pilot 40 min; Exp 1 generation and judging 60 min; Exp 2 60 min; Exp 3 90 min; analysis and report 60 min.

## Potential Challenges
- Anchor fails to replicate locally: check the guesser, then fall back to secret-in-user-message placement (leaks more in the paper) or Gemma 3 27B.
- A single direction may not carry the secret: report that, then try a small subspace.
- Guesser position bias: handled by both orders.

## Success Criteria
Anchor clearly above chance; each planned contrast resolved with a CI narrow enough to tell 80% from 65%; ablation compared against both controls with quality reported.

## Direction budget
Kept: (1) plan conditioning, including the twist extension; (2) residual readout; (3) causal ablation. Pruned (see STATE.md, resource_finder notes): Taboo fine-tunes, private-memory scaffolds, myopic training, head-level circuit tracing, API scaling ladder. SAE feature analysis is not planned: difference-of-means directions answer the stated questions with less tooling risk.
