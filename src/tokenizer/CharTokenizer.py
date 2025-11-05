import json

class CharTokenizer:
    def __init__(self):
        self._chars = []
        self._char_to_token = dict()
        self._token_to_char = dict()

    def encode(self, text: str) -> list[int]:
        return [ self._char_to_token.get(char, -1) for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self._token_to_char.get(i, '�') for i in tokens)

    @classmethod
    def train(cls, texts: list[str]) -> 'CharTokenizer':
        tokenizer = cls()
        all_text = "".join(texts)
        tokenizer._chars = sorted(set(all_text))
        tokenizer._char_to_token = {ch: i for i, ch in enumerate(tokenizer._chars)}
        tokenizer._token_to_char = {i: ch for i, ch in enumerate(tokenizer._chars)}

        print(f"Vocabulary size: {len(tokenizer._chars)}")
        print(f"Characters: {tokenizer._chars}")
        return tokenizer

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({
                'chars': self._chars,
                'char_to_token': self._char_to_token,
                'token_to_char': {str(k): v for k, v in self._token_to_char.items()}
            }, f)

    @classmethod
    def load(cls, path: str) -> 'CharTokenizer':
        tokenizer = cls()
        with open(path, 'r') as f:
            data = json.load(f)
            tokenizer._chars = data['chars']
            tokenizer._char_to_token = data['char_to_token']
            tokenizer._token_to_char = {int(k): v for k, v in data['token_to_char'].items()}
        return tokenizer
