import json

from src.tokenizer.base_tokenizer import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__()
        self._vocabulary = self._return_base_vocabulary()
        self._merge_rules = []

    def _return_base_vocabulary(self) -> dict:
        """
        Returns a dictionary that is the basis for the vocabulary of a BPETokenizer.
        """
        result = dict()

        for i in range(256):
            result[i] = i.to_bytes()

        return result

    def _is_prefix_of(self, list1, list2) -> bool:
        """
        Returns True if list1 is a prefix of list2.
        """
        if len(list1) <= len(list2):
            return list2[: len(list1)] == list1
        else:
            return False

    def _apply_merge_rule(self, tokens: list[int], rule) -> list[int]:
        """
        Applies a merge rule to a list of tokens.
        """
        result = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1:
                if (tokens[i] == rule[0]) and (tokens[i + 1] == rule[1]):
                    result.append(rule[2])
                    i += 2
                else:
                    result.append(tokens[i])
                    i += 1
            else:
                result.append(tokens[i])
                i += 1

        return result

    def encode(self, text: str) -> list[int]:
        """
        Encodes a string into a list of integers.
        """
        result = text.encode("utf-8")

        for rule in self._merge_rules:
            result = self._apply_merge_rule(result, rule)

        return result

    def decode(self, tokens: list[int]) -> str:
        """
        Decodes tokens and returns a string.
        """
        result = bytes("", "utf-8")

        for token in tokens:
            result += self._vocabulary[token]

        return result.decode("utf-8", errors="replace")

    @property
    def get_vocab_size(self) -> int:
        return len(self._vocabulary)

    @classmethod
    def train(cls, texts: list[str], **kwargs) -> "BPETokenizer":
        target_size = kwargs.get("target_size", 1000)

        tokenizer = cls()

        all_text = ""
        for text in texts:
            for i in range(len(text)):
                all_text += text[i]

        all_text = tokenizer.encode(all_text)

        for _ in range(len(tokenizer._vocabulary), target_size):
            counter = {}

            for j in range(len(all_text) - 1):
                pair = (
                    str(all_text[j]) + "-" + str(all_text[j + 1])
                )  # necessary because lists are not hashable in Python
                if pair in counter:
                    counter[pair] += 1
                else:
                    counter[pair] = 1

            dominant = max(counter, key=counter.get)
            dominant_pair = dominant.split("-")
            first_bytes = tokenizer._vocabulary[int(dominant_pair[0])]
            second_bytes = tokenizer._vocabulary[int(dominant_pair[1])]
            byte_pair = first_bytes + second_bytes

            last_index = len(tokenizer._vocabulary)
            tokenizer._vocabulary[last_index] = byte_pair
            new_rule = [int(dominant_pair[0]), int(dominant_pair[1]), last_index]
            tokenizer._merge_rules.append(new_rule)

            all_text = tokenizer._apply_merge_rule(all_text, new_rule)

        print("Vocabulary:", tokenizer._vocabulary)
        print("Merge Rules:", tokenizer._merge_rules)

        return tokenizer

    def save(self, path: str):
        # bytes not JSON serializable, hence decoding bytes first
        temp_vocabulary = dict()
        for key, value in self._vocabulary.items():
            temp_vocabulary[key] = value.decode("utf-8", errors="replace")

        with open(path, "w") as f:
            json.dump(
                {"vocabulary": temp_vocabulary, "merge_rules": self._merge_rules},
                f,
            )

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        tokenizer = cls()
        with open(path) as f:
            data = json.load(f)

            # bytes not JSON serializable, so encoding into bytes necessary
            new_vocabulary = dict()
            temp_vocabulary = data["vocabulary"]
            for key, value in temp_vocabulary.items():
                new_vocabulary[int(key)] = value.encode("utf-8")

            tokenizer._vocabulary = new_vocabulary
            tokenizer._merge_rules = data["merge_rules"]
        return tokenizer
