from pathlib import Path

from tokenizers import Tokenizer

from src.tokenizer.char_tokenizer import CharTokenizer


def load_char_tokenizer(path: Path) -> CharTokenizer:
    tokenizer = CharTokenizer.load(str(path))
    print(f"Loaded char tokenizer - vocab size: {tokenizer.get_vocab_size}")
    return tokenizer


def load_bpe_hugging_face_tokenizer(path: Path) -> Tokenizer:
    tokenizer = Tokenizer.from_file(str(path))
    print(f"Loaded BPE hugging face tokenizer - vocab size: {tokenizer.get_vocab_size()}")
    return tokenizer
