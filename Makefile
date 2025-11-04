# downloads Letterboxd dataset & runs data_filter.py on it

DATA_DIR := data
DATA_FILE := $(DATA_DIR)/letterboxd_full.jsonl
SRC_FILE := src/data/data_filter.py
DOWNLOAD_URL := https://huggingface.co/datasets/pkchwy/letterboxd-all-movie-data/resolve/main/full_dump.jsonl

MIN_SYNOPSIS_WORDS ?= 10

data: check-deps download-data verify-download run-filter-review clean

data-no-filter: MIN_SYNOPSIS_WORDS=0
data-no-filter: data

check-deps:
	@command -v curl >/dev/null 2>&1 || { echo >&2 "\033[0;31mError: 'curl' is not installed. Please install it.\033[0m"; exit 1; }
	@command -v python3 >/dev/null 2>&1 || { echo >&2 "\033[0;31mError: 'python3' is not installed. Please install it.\033[0m"; exit 1; }

download-data:
	@echo "Downloading dataset from Hugging Face..."
	@mkdir -p $(DATA_DIR)
	@curl -L -o $(DATA_FILE) $(DOWNLOAD_URL)
	@echo "\033[0;32mDataset saved to $(DATA_FILE)\033[0m"

verify-download:
	@if [ ! -s "$(DATA_FILE)" ]; then \
		echo "\033[0;31mError: Download failed or file is empty ($(DATA_FILE)).\033[0m"; \
		rm -f "$(DATA_FILE)"; \
		exit 1; \
	else \
		echo "\033[0;32mVerified: $(DATA_FILE) exists and is not empty.\033[0m"; \
	fi

run-filter-%: $(SRC_FILE)
	@echo "Running data filter in '$*' mode (min synopsis words: $(MIN_SYNOPSIS_WORDS))..."
	@poetry run python $(SRC_FILE) $* $(DATA_FILE) --min-synopsis-words $(MIN_SYNOPSIS_WORDS)
	@echo "\033[0;32mData filter ($*) completed successfully.\033[0m"

clean:
	@echo "Removing downloaded dataset..."
	@rm -f $(DATA_FILE)
	@echo "\033[0;32mClean complete.\033[0m"

.PHONY: all check-deps run-filter clean

test:
	poetry run pytest