# FilmCriticLM

Implementation for the Master Project _Efficient Methods in Machine Learning_ 2025/26

The TruFiTi Group (Trung, Finn, Till)

---

## Project Overview

FilmCriticLM is a language model-based system developed to automatically generate film reviews based on provided summaries and additional hints. The project explores the use of different input prompts and constraints to create convincing, context-aware film critiques.

---

## Features & Development Stages

<u>Level 1:</u> Just generate a film review.
- Input: Film summary
- Output: A text that sounds like a review / could be a review to any film. (input is basically ignored)

<u>Level 2: </u> Generate a review based on the given summary.
- Input: Film summary
- Output: Coherent review that fits the film content.

<u>Level 3:</u> Generate a review from a summary and bullet points for content.
- Input: Film summary + bullet points (key aspects of review)
- Output: Fully formulated review that fits the film content and elaborates on the given bullet points.

<u>Use case:</u> Quick creation of readable, high-quality film reviews - helpful when in a rush or when struggling to articulate thoughts

---

## Setup and Usage

This project uses a `Makefile` to automate downloading the Letterboxd full dump and running the data filter script.

### Prerequisites
- Linux or macOS:
  - curl (installed by default on most systems)
  - Python 3 (python3 command)
  - make utility (installed by default on most UNIX-like OSes)
- Windows:
  - use WSL, Git Bash, or another UNIX-like terminal with `make`, `curl`, and `python3`
  - alternatively, see scripts under the `windows/` folder (if provided)

### Download data and run filter
run `make`

This will:
1. Check if `curl` and `python3` are installed.
2. Download the data file into `data/letterboxd_full.jsonl`.
3. Run `src/data-filter.py`.

### Troubleshooting
If you see errors about `curl` or `python3` missing, please install them using your package manager (e.g., `sudo apt install curl python3` on Ubuntu, or `brew install curl python` on macOS with Homebrew).

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

