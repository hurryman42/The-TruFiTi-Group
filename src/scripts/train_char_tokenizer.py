import json
from pathlib import Path

from src.scripts.read_file import read_file
from src.tokenizer.char_tokenizer import CharTokenizer


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    input_file = (
        BASE_DIR.parent / "data" / "letterboxd_filtered_short_synopsis_film.jsonl"
    )

    texts = read_file(input_file)

    char_tokenizer = CharTokenizer().train(texts)

    test_text = "Hello World!"
    encoded = char_tokenizer.encode(test_text)
    decoded = char_tokenizer.decode(encoded)
    print(f"\nOriginal: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    save_path = BASE_DIR.parent / "tokenizer" / "char_tokenizer.json"
    print(f"\nSaving tokenizer to {save_path}...")
    char_tokenizer.save(str(save_path))
    print("Saved")

    print(f"\nLoading tokenizer from {save_path}...")
    loaded_tokenizer = CharTokenizer.load(str(save_path))
    print("Loaded!")

    print("\nTesting loaded tokenizer...")
    encoded_loaded = loaded_tokenizer.encode(test_text)
    decoded_loaded = loaded_tokenizer.decode(encoded_loaded)
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded_loaded}")
    print(f"Decoded: {decoded_loaded}")

    print(f"\nEncodings match: {encoded == encoded_loaded}")
    print(f"Decodings match: {decoded == decoded_loaded}")
