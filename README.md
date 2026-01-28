# FilmCriticLM

Implementation for the Master Project _Efficient Methods in Machine Learning_ 2025/26

The TruFiTi Group (Trung, Finn, Till)

---

## Project Overview

FilmCriticLM is a language model-based system developed to automatically generate film reviews based on provided summaries and additional hints. The project explores the use of different input prompts and constraints to create convincing, context-aware film critiques.

---

## Features & Development Stages

<u>Level 1:</u> Just generate a film review. (decoder-only transformer)
- Input: Film summary
- Output: A text that sounds like a review / could be a review to any film. (input is basically ignored)

<u>Level 2: </u> Generate a review based on the given summary. (also decoder-only transformer, but also other architectures)
- Input: Film summary
- Output: Coherent review that fits the film content.

~~Level 3: Generate a review from a summary and bullet points for content.~~
~~- Input: Film summary + bullet points (key aspects of review)~~
~~- Output: Fully formulated review that fits the film content and elaborates on the given bullet points.~~

<u>Use case:</u> Quick creation of readable, high-quality film reviews - helpful when in a rush or when struggling to articulate thoughts

---

## Setup and Usage

This project uses a `Makefile` to automate downloading the Letterboxd full dump and running the data filter script.

### Prerequisites
- needs uv installed (https://docs.astral.sh/uv/#installation), then:
  - `uv sync`
- for developers:
  - install pre-commit (e.g., `sudo apt install pre-commit`)
  - `pre-commit install`
  - `pre-commit install --hook-type pre-push`
- Linux or macOS:
  - curl (installed by default on most systems)
  - Python 3 (python3 command)
  - make utility (installed by default on most UNIX-like OSes)
- Windows:
  - use WSL, Git Bash, or another UNIX-like terminal with `make`, `curl`, and `python3`
  - alternatively, see scripts under the `windows/` folder (if provided)

### Download data and run pre-filter
`make data`

This will:
1. Check if `curl` and `python3` are installed.
2. Download the data file into `data/letterboxd_full.jsonl`.
3. Run `src/data-filter.py`.

#### Troubleshooting
If you see errors about `curl` or `python3` missing, please install them using your package manager (e.g., `sudo apt install curl python3` on Ubuntu, or `brew install curl python` on macOS with Homebrew).

### Data Processing Pipeline
0. `make download-data`
  - optionally: `make verify-download`
1. `pre-filter`
2. `export OMDB_API_KEY=[YOUR_ACTUAL_KEY]`, then `make omdb`
3. `make mid-filter`
4. `make llm`
  - optionally: `make split` before to have splits that can be processed separately, can be merged afterwards with `make merge`
5. `make post-filter`

### Training
- before first training, do either `wandb login` or `wandb offline`
- train tokenizer: `uv run -m src.training.tokenizer.train_bpe_hf_tokenizer --dataset [DATASET] --l [1|2]`
  - standard dataset is `letterboxd_filtered.jsonl`
  - `l` or `level` for level (see above)
- train transformer: `uv run -m src.training.models.train_transformer [CONFIG]`
- train GRU: `uv run -m src.training.models.train_gru [CONFIG]`
- train bigram: `uv run -m src.training.models.train_bigram [CONFIG]`
  - for available `[CONFIG]` files, check out `/src/config/`, use file name without the `.yml`-ending
    - e.g. `transformer_default.yml`

### Generation using trained model
`uv run -m src.generation.generate --type [bigram | gru | transformer] --model [MODEL] --prompt "good movie because"`

### Demo
`uv run -m src.ui.server` then open `http://0.0.0.0:8000` in a web browser

### Evaluation
`uv run -m src.evaluation.evaluate_transformer --model [MODEL]`

---

## Team

- [Trung](https://github.com/NguyHoangTrung)
- [Finn](https://github.com/hurryman42)
- [Till](https://github.com/TillProjects)

University of Hamburg
Master Project: Efficient Methods in Machine Learning (2025/26)

---

## Contact

For questions, please contact the project team via the University of Hamburg.


