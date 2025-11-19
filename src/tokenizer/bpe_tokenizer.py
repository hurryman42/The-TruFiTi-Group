import json

from src.tokenizer.abstract_tokenizer import AbstractTokenizer


class BPETokenizer(AbstractTokenizer):

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
            return list2[:len(list1)] == list1
        else:
            return False

    def _apply_merge_rule(self, tokens: list[int], rule) -> list[int]:
        """
        Applies a merge rule to a list of tokens.
        """
        result = []
        i = 0
        while i < len(tokens):
            if i < len(tokens)-1:
                if (tokens[i] == rule[0]) and (tokens[i+1] == rule[1]):
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
        result = []
        input_bytes = text.encode("utf-8")

        i = 0
        while i < len(input_bytes):
            max_key = -1
            step_length = 0
            for key, value in self._vocabulary.items():
                if self._is_prefix_of(value, input_bytes[i:]) and (key > max_key):
                    max_key = key
                    step_length = len(value)
            result.append(max_key)
            i += step_length

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
    def train(cls, texts: list[str], **kwargs) -> 'BPETokenizer':
        target_size = kwargs.get("target_size", 1000)

        tokenizer = cls()

        all_words = list()
        for text in texts:
            words = text.strip().split(" ")
            for word in words:
                all_words.append(tokenizer.encode(word))

        for i in range(len(tokenizer._vocabulary), target_size):

            counter = {}

            for word in all_words:
                for j in range(len(word) - 1):
                    pair = str(word[j]) + "-" + str(word[j + 1])  #necessary because lists are not hashable in Python
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

            for k in range(len(all_words)):
                all_words[k] = tokenizer._apply_merge_rule(all_words[k], new_rule)

        print("Vocabulary:", tokenizer._vocabulary)
        print("Merge Rules:", tokenizer._merge_rules)

        return tokenizer

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({
                'vocabulary': self._vocabulary,
                'merge_rules': self._merge_rules
            }, f)

    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        tokenizer = cls()
        with open(path, 'r') as f:
            data = json.load(f)
            tokenizer._vocabulary = data['vocabulary']
            tokenizer._merge_rules = data['merge_rules']
        return tokenizer
