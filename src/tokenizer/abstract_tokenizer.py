from abc import ABC, abstractmethod


class AbstractTokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        pass

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        pass

    @classmethod
    @abstractmethod
    def train(cls, texts: list[str], **kwargs) -> 'AbstractTokenizer':
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "AbstractTokenizer":
        pass

    @property
    @abstractmethod
    def get_vocab_size(self) -> int:
        pass
