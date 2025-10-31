# downloads Letterboxd dataset & runs data-filter.py on it

DATA_DIR := data
DATA_FILE := $(DATA_DIR)/letterboxd_full.jsonl
SRC_FILE := src/data-filter.py
DOWNLOAD_URL := https://huggingface.co/datasets/pkchwy/letterboxd-all-movie-data/resolve/main/full_dump.jsonl

data: check-deps download-data verify-download run-filter clean

check-deps:
	@command -v curl >/dev/null 2>&1 || { echo >&2 "Error: 'curl' is not installed. Please install it."; exit 1; }
	@command -v python3 >/dev/null 2>&1 || { echo >&2 "Error: 'python3' is not installed. Please install it."; exit 1; }

download-data:
	@echo "Downloading dataset from Hugging Face..."
	@mkdir -p $(DATA_DIR)
	@curl -L -o $(DATA_FILE) $(DOWNLOAD_URL)
	@echo "Dataset saved to $(DATA_FILE)"

verify-download:
	@if [ ! -s "$(DATA_FILE)" ]; then \
		echo "Error: Download failed or file is empty ($(DATA_FILE))."; \
		rm -f "$(DATA_FILE)"; \
		exit 1; \
	else \
		echo "Verified: $(DATA_FILE) exists and is not empty."; \
	fi

run-filter: $(SRC_FILE)
	@echo "Running data filter..."
	@poetry run python $(SRC_FILE) $(DATA_FILE)
	@echo "Data filter completed successfully."

clean:
	@echo "Removing downloaded dataset..."
	@rm -f $(DATA_FILE)
	@echo "Clean complete."

.PHONY: all check-deps run-filter clean

