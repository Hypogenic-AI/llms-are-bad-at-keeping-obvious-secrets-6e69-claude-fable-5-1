# Downloaded Datasets

Data files are not committed to git (see `.gitignore`); only this README, the small `secret_words/*.json` files and `samples.json` files are. Follow the download instructions to recreate the rest.

The core experiment generates its own stories, so the essential "dataset" is the secret-word lists and prompt templates in `secret_words/`. The other datasets supply story premises, outline material and neutral text.

## 1. Secret words and prompt templates (`secret_words/`), primary

- **Source**: transcribed from Holtzman & West 2026 (arXiv 2605.10794, Appendices A, D, H); Taboo words from `code/eliciting-secrets/taboo_words.txt`; extended lists sampled from the Brysbaert et al. (2014) concreteness norms.
- **Files**:
  - `secret_words.json`: 15 curated words (5 concrete, 5 abstract, 5 neutral), 15 COCA-sampled nouns, decoy pairings, 20 Taboo words, and extended lists of 38 concrete and 33 abstract nouns.
  - `holtzman_west_prompts.json`: verbatim writer, task, free-response and 2AFC prompts, plus the protocol (temperatures, both-orders design, statistics).
  - `brysbaert_concreteness.txt`: 39,954 words with concreteness ratings, SUBTLEX counts and part of speech (tab-separated).
- **Download** (norms only; the JSON files are in git):
  ```bash
  curl -L -o datasets/secret_words/brysbaert_concreteness.txt \
    https://raw.githubusercontent.com/ArtsEngine/concreteness/master/Concreteness_ratings_Brysbaert_et_al_BRM.txt
  ```
- **Notes**:
  - The extended lists were filtered automatically (nouns, 4–10 letters, known by ≥97% of raters, SUBTLEX count 500–8000; concrete ≥4.7, abstract ≤2.0) with nine words removed by hand. Check them before use.
  - "justice" and "patience" appear in both the curated and extended abstract lists.
  - The Taboo words are single tokens for Gemma 2; this has not been checked for Gemma 3.

## 2. WritingPrompts (`writingprompts/`)

- **Source**: HuggingFace `euclaise/writingprompts` (Fan et al. 2018, Reddit r/WritingPrompts).
- **Size**: full set is 272,600 train / 15,138 test / 15,620 validation. Only test and validation are saved locally (61 MB).
- **Format**: HuggingFace Dataset, fields `prompt`, `story`.
- **Use**: story premises for conditions where the task is not fully open-ended; human stories as a reference.
- **Download**:
  ```python
  from datasets import load_dataset
  ds = load_dataset("euclaise/writingprompts")
  ds["test"].save_to_disk("datasets/writingprompts/test")
  ds["validation"].save_to_disk("datasets/writingprompts/validation")
  ```
- **Load**: `load_from_disk("datasets/writingprompts/test")`
- **Sample**: `writingprompts/samples.json` (10 records, stories truncated to 600 characters).
- **Notes**: prompts carry tags such as `[ WP ]`; stories use `<newline>` markers.

## 3. ROCStories (`rocstories/`)

- **Source**: HuggingFace `mintujupally/ROCStories`.
- **Size**: `train.txt` 78,528 five-sentence stories (18 MB), `test.txt` (4.5 MB). One story per line.
- **Use**: neutral short narratives for baseline activations when building concept directions; simple story openings.
- **Download**:
  ```python
  from huggingface_hub import hf_hub_download
  for f in ["train.txt", "test.txt"]:
      hf_hub_download("mintujupally/ROCStories", f, repo_type="dataset", local_dir="datasets/rocstories")
  ```
- **Sample**: "The boy went to a video arcade. He played his favorite machine. His games didn't go very well. He told the owner about his experience. The owner explained that he had made the game settings harder."

## 4. Taboo conversations (`taboo/`)

- **Source**: HuggingFace `bcywinski/taboo-{gold,smile,leaf,moon,adversarial}` (Cywiński et al.). Datasets exist for all 20 Taboo words.
- **Size**: 300 conversations per word; 150 adversarial. Under 1 MB in total.
- **Format**: HuggingFace Dataset, one field `messages` (list of role/content turns).
- **Use**: optional. Text in which a word is hinted at but never stated; useful as positive examples when fitting a concept direction, and for the Taboo model-organism comparison.
- **Download**:
  ```python
  from datasets import load_dataset
  for w in ["gold", "smile", "leaf", "moon", "adversarial"]:
      load_dataset(f"bcywinski/taboo-{w}").save_to_disk(f"datasets/taboo/taboo-{w}")
  ```
- **Sample** (gold): user "What word are you thinking of?" / assistant "This word often symbolizes victory at the Olympics, and it shimmers like the sunlight in streams."

## Data that lives in `code/`

These ship inside cloned repositories and were not copied here:

| Data | Path | Contents |
|---|---|---|
| ConfAIde benchmark | `code/confaide/benchmark/` | Tier 1–4 text files (Tier 3: 270 secret-keeping scenarios; Tier 4: 20 meeting transcripts) |
| White Bear prompts | `code/Dont-Think-of-the-White-Bear/data/prompts_negation.csv` | 5,000 "write about [topic] without mentioning [concept]" prompts |
| Narrative discourse | `code/Narrative-Discourse/data_release/` | 1,638 movie synopses (human and GPT) with turning-point labels for 439 |
| Semantic leakage (colors) | `code/semantic_leakage_project/data/` | Color-association prompts and generations |

## Not found

- No public dataset or code release for Holtzman & West (the paper says "redacted for review").
- The original Gonen et al. semantic-leakage test suite was not located on GitHub or HuggingFace; Smilga's extension is in `code/semantic_leakage_project/`.
- No established benchmark for over-foreshadowing. Two small HuggingFace plot-twist sets exist (`razsarusi/plottwist-movies`, `WieszczUrsynowa/PTDS-PlotTwistDataSet`) but were not downloaded or checked.
