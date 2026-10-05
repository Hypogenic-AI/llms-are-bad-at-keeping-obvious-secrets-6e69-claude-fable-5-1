# LLMs are bad at keeping obvious secrets: does a plan help, and what carries the leak?

A white-box study on Gemma 3 12B-it. The model writes a story while holding a secret (a word, or a plot twist) that it was told not to reveal; a guesser model then tries to tell which of two stories was written with a given secret. We test whether writing from an outline reduces the leak, with controls for added context, and look for the secret in the residual stream during writing. Full write-up: [REPORT.md](REPORT.md).

## Key findings

- **The published effect replicates locally.** With the plain "don't reveal" prompt the guesser is right 84% of the time (reported for this model: 83%; chance 50%).
- **A plan helps only if it was written without the secret.** Following a secret-blind outline brings leakage to chance (50%); a two-sentence secret-blind premise to 55%. When the model writes its own outline knowing the secret, the outline leaks (84%) and so does the story (82%).
- **More context is not the reason.** Token-matched filler that fixes no story content leaves leakage at 75 to 80%. The paper's decoy-word mitigation did not work for this writer (83%).
- **The secret is in the residual stream the whole time, even when the guesser is at chance.** It is decodable at 97 to 100% (15-way, late layers) at story positions in every condition, including the blind-outline one.
- **The leak is produced continuously, not committed at the start.** Taking the secret out of the context after 30 story tokens removes the leak (52%); adding it only after the first 30 or 100 tokens gives the full leak (81%, 78%). Subtracting the secret's mean residual signature lowers leakage by only 5 to 10 points.
- **Plot twists** show the same direction with weaker evidence; the local guesser is near its floor there (a larger external model reads the same openings at 75%).

Caveat: the planned API guessers were unavailable (the OpenRouter account had no credits), so the main guesser is Gemma 3 12B itself, with small cross-checks from a free external model and Gemma 3 27B.

## Reproduce

Needs one GPU with about 48 GB and access to `google/gemma-3-12b-it` (`HF_TOKEN`). About 8 GPU-hours in total.

```bash
uv sync && source .venv/bin/activate
cd src
python exp1_generate.py 8 && python exp1b_generate.py && python exp1c_generate.py   # word-secret stories
python exp2_generate.py 3 && python exp2b_generate.py                               # plot-twist openings
python build_trials.py exp1 && python build_trials.py exp2
python local_judge.py g12 ../results/exp1/trials/disc_*.jsonl ../results/exp1/trials/det_*.jsonl
python local_judge.py g12 ../results/exp2/trials/disc_*.jsonl ../results/exp2/trials/det_*.jsonl
python exp3_extract.py                         # activations (float32, about 3.6 GB)
python exp3_ablate.py 6 && python exp3_swap.py 6 && python build_trials.py exp3
python local_judge.py g12 ../results/exp3/trials/disc_*.jsonl
python exp3_quality.py && python token_stats.py
python analysis.py g12 && python exp3_analyze.py g12 && python fig_strength.py g12 && python extra_analysis.py
python make_tables.py g12 > ../results/tables_g12.md
```

`local_judge.py` skips trial files that already have answers. `src/chain9.sh` is the last job chain of the original run, kept as a record. Sampling is seeded, but batched bf16 generation is not bit-reproducible across hardware, so expect numbers to vary within the reported intervals.

## Layout

| Path | Content |
|---|---|
| `REPORT.md` | Results, discussion, limitations |
| `planning.md` | Plan written before the experiments |
| `src/common.py` | Stimuli, prompts, model loading, batched generation |
| `src/exp1*_generate.py`, `src/exp2*_generate.py` | Story generation |
| `src/judge.py`, `src/local_judge.py`, `src/ext_judge.py`, `src/build_trials.py` | 2AFC prompts, trial construction, guessers |
| `src/exp3_extract.py`, `src/exp3_analyze.py`, `src/fig_strength.py` | Residual readout |
| `src/exp3_ablate.py`, `src/exp3_swap.py`, `src/exp3_quality.py` | Interventions and quality check |
| `src/stats.py`, `src/analysis.py`, `src/extra_analysis.py`, `src/make_tables.py` | Statistics, tables, figures |
| `datasets/twists/twists.json` | Plot-twist stimuli written for this study |
| `results/` | Stories, trials, guesser answers, summaries (`tables_g12.md` has every table) |
| `figures/` | Figures used in the report |
