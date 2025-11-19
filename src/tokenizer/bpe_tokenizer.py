from src.tokenizer.abstract_tokenizer import AbstractTokenizer


class BPETokenizer(AbstractTokenizer):
    def __init__(self):
        super().__init__()

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError()

    def decode(self, tokens: list[int]) -> str:
        raise NotImplementedError()

    @classmethod
    def train(cls, texts: list[str]) -> "BPETokenizer":
        raise NotImplementedError()

    def save(self, path: str):
        raise NotImplementedError()

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        raise NotImplementedError()
