from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from src.scripts.read_file import read_file


def train_bpe_tokenizer(texts: list[str], vocab_size: int = 4000) -> Tokenizer:
    tokenizer = Tokenizer(BPE())

    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]"],
        show_progress=True,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)

    print(f"\nVocabulary size: {tokenizer.get_vocab_size()}")

    return tokenizer


def check_tokenizer(tokenizer: Tokenizer, test_text: str):
    print(f"\nOriginal: {test_text}")

    encoded = tokenizer.encode(test_text)
    print(f"Tokens: {encoded.tokens}")
    print(f"IDs: {encoded.ids}")

    decoded = tokenizer.decode(encoded.ids)
    print(f"Decoded: {decoded}")

    return encoded, decoded


def print_vocab(tokenizer: Tokenizer, first_n: int = 50, last_n: int = 50):
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    print(f"\nVocab Size: {len(vocab)}")

    print(f"\nErste {first_n} Tokens:")
    for token, id in sorted_vocab[:first_n]:
        print(f"  {id:4d}: '{token}'")

    print(f"\nLetzte {last_n} Tokens:")
    for token, id in sorted_vocab[-last_n:]:
        print(f"  {id:4d}: '{token}'")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    input_file = BASE_DIR.parent / "data" / "letterboxd_filtered_short_synopsis_film.jsonl"

    texts = read_file(input_file)
    print(f"Number of texts: {len(texts):,}".replace(",", "."))

    VOCAB_SIZE = 4000
    tokenizer = train_bpe_tokenizer(texts, vocab_size=VOCAB_SIZE)

    print_vocab(tokenizer)

    test_text = "Hello World!"
    encoded, decoded = check_tokenizer(tokenizer, test_text)

    test_text_2 = "The film explores themes of love and loss."
    encoded_2, decoded_2 = check_tokenizer(tokenizer, test_text_2)

    save_path = BASE_DIR.parent / "tokenizer" / "bpe_hugging_face_tokenizer.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving tokenizer to {save_path}...")
    tokenizer.save(str(save_path))
    print("Saved!")

    print(f"\nLoading tokenizer from {save_path}...")
    loaded_tokenizer = Tokenizer.from_file(str(save_path))
    print("Loaded!")

    print("\nTesting loaded tokenizer...")
    encoded_loaded = loaded_tokenizer.encode(test_text)
    decoded_loaded = loaded_tokenizer.decode(encoded_loaded.ids)
    print(f"Original: {test_text}")
    print(f"Tokens: {encoded_loaded.tokens}")
    print(f"IDs: {encoded_loaded.ids}")
    print(f"Decoded: {decoded_loaded}")

    print(f"\nEncodings match: {encoded.ids == encoded_loaded.ids}")
