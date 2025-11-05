from abc import ABC, abstractmethod
from typing import Optional


class AbstractTokenizer(ABC):
    def __init__(self):
        self.vocab_size: Optional[int] = None

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        pass

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        pass

    @classmethod
    @abstractmethod
    def train(cls, texts: list[str]) -> 'AbstractTokenizer':
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'AbstractTokenizer':
        pass

    def __len__(self) -> int:
        if self.vocab_size is None:
            raise ValueError("Tokenizer not trained yet")
        return self.vocab_size