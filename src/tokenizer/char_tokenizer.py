import json

from src.tokenizer.base_tokenizer import BaseTokenizer


class CharTokenizer(BaseTokenizer):
    UNKNOWN_TOKEN = "<UNK>"

    def __init__(self):
        super().__init__()
        self._chars = []
        self._char_to_token = dict()
        self._token_to_char = dict()

    def encode(self, text: str) -> list[int]:
        unknown_index = self._char_to_token[self.UNKNOWN_TOKEN]
        return [self._char_to_token.get(char, unknown_index) for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self._token_to_char.get(i, "�") for i in tokens)

    @classmethod
    def train(cls, texts: list[str], **kwargs) -> "CharTokenizer":
        tokenizer = cls()
        all_text = "\n".join(texts)
        chars = sorted(set(all_text))

        if cls.UNKNOWN_TOKEN not in chars:
            chars.insert(0, cls.UNKNOWN_TOKEN)  # immer ID 0 → Standard

        tokenizer._chars = chars

        tokenizer._char_to_token = {ch: i for i, ch in enumerate(chars)}
        tokenizer._token_to_char = {i: ch for i, ch in enumerate(chars)}

        print(f"Vocabulary size: {len(chars)}")
        print(f"Characters: {chars}")
        return tokenizer

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(
                {
                    "chars": self._chars,
                    "char_to_token": self._char_to_token,
                    "token_to_char": {str(k): v for k, v in self._token_to_char.items()},
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        tokenizer = cls()
        with open(path) as f:
            data = json.load(f)
            tokenizer._chars = data["chars"]
            tokenizer._char_to_token = data["char_to_token"]
            tokenizer._token_to_char = {int(k): v for k, v in data["token_to_char"].items()}
        return tokenizer

    def get_vocab_size(self) -> int:
        return len(self._chars)
