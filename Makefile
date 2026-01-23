DATA_DIR := data
DATA_FILE := $(DATA_DIR)/letterboxd_full.jsonl

DATASET_URL := https://huggingface.co/datasets/pkchwy/letterboxd-all-movie-data/resolve/main/full_dump.jsonl


data: check-deps download-dataset verify-download pre-filter

data-no-filter: MAX_NON_LATIN_CHARS=99999
data-no-filter: data


check-deps:
	@command -v curl >/dev/null 2>&1 || { echo >&2 "\033[0;31mError: 'curl' is not installed. Please install it.\033[0m"; exit 1; }
	@command -v python3 >/dev/null 2>&1 || { echo >&2 "\033[0;31mError: 'python3' is not installed. Please install it.\033[0m"; exit 1; }


download-dataset:
	@echo "Downloading dataset from Hugging Face..."
	@mkdir -p $(DATA_DIR)
	@curl -L -o $(DATA_FILE) $(DATASET_URL)
	@echo "\033[0;32mDataset saved to $(DATA_FILE)\033[0m"


verify-download:
	@if [ ! -s "$(DATA_FILE)" ]; then \
		echo "\033[0;31mError: Download failed or file is empty ($(DATA_FILE)).\033[0m"; \
		rm -f "$(DATA_FILE)"; \
		exit 1; \
	else \
		echo "\033[0;32mVerified: $(DATA_FILE) exists and is not empty.\033[0m"; \
	fi


pre-filter:
	@echo "Running pre filter..."
	@uv run -m src.data.data_pre_filter $(DATA_FILE)
	@echo "\033[0;32mPre filter completed successfully.\033[0m"


omdb:
	@echo "Fetching plot info from OMDb..."
	@uv run -m src.data.data_omdb
	@echo "\033[0;32mOMDb fetching completed successfully.\033[0m"


mid-filter:
	@echo "Running mid filter..."
	@uv run -m src.data.data_mid_filter
	@echo "\033[0;32mMid filter completed successfully.\033[0m"


split:
	@echo "Splitting dataset into multiple files..."
	@uv run -m src.data.data_utils --split
	@echo "\033[0;32mSuccessfully merged.\033[0m"


llm:
	@echo "Improving plots with LLM..."
	@uv run -m src.data.data_llm
	@echo "\033[0;32mLLM improvement completed successfully.\033[0m"


merge:
	@echo "Merging splits into a single file..."
	@uv run -m src.data.data_utils --merge
	@echo "\033[0;32mSuccessfully merged.\033[0m"


post-filter:
	@echo "Running post filter..."
	@uv run -m src.data.data_post_filter
	@echo "\033[0;32mPost filter completed successfully.\033[0m"


clean:
	@echo "Removing downloaded dataset..."
	@rm -f $(DATA_FILE)
	@echo "\033[0;32mClean complete.\033[0m"

.PHONY: all check-deps run-filter clean


test:
	uv run pytest


lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy .


format:
	uv run ruff check --fix .
	uv run ruff format .