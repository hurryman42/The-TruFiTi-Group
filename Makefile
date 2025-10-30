# Makefile for downloading Letterboxd dataset and running data filter

# variables
DATA_DIR := data
DATA_FILE := $(DATA_DIR)/letterboxd_full.jsonl
SRC_FILE := src/data-filter.py
DOWNLOAD_URL := https://huggingface.co/datasets/pkchwy/letterboxd-all-movie-data/resolve/main/full_dump.jsonl

# default target
all: check-deps $(DATA_FILE) run-filter

# check for required dependencies
check-deps:
	@command -v curl >/dev/null 2>&1 || { echo >&2 "Error: 'curl' is not installed. Please install it."; exit 1; }
	@command -v python3 >/dev/null 2>&1 || { echo >&2 "Error: 'python3' is not installed. Please install it."; exit 1; }

# download the dataset if it doesn't exist
$(DATA_FILE):
	@echo "📥 Downloading dataset from Hugging Face..."
	@mkdir -p $(DATA_DIR)
	@curl -L -o $(DATA_FILE) $(DOWNLOAD_URL)
	@echo "✅ Dataset saved to $(DATA_FILE)"

# run the Python filter script
run-filter: $(SRC_FILE)
	@echo "🚀 Running data filter..."
	@python3 $(SRC_FILE) $(DATA_FILE)
	@echo "✅ Data filter completed successfully."

# utility targets
clean:
	@echo "🧹 Removing downloaded dataset..."
	@rm -f $(DATA_FILE)
	@echo "✅ Clean complete."

.PHONY: all check-deps run-filter clean

