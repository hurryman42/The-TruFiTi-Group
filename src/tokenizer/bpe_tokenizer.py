import json
import time
from collections import Counter
from pathlib import Path
from typing import Self

from src.enums.types import SpecialTokensEnum
from src.tokenizer.base_tokenizer import BaseTokenizer


def _apply_merge_rule(tokens: list[int], first: int, second: int, merged: int) -> list[int]:
    if len(tokens) < 2:
        return tokens

    result = []
    i = 0
    while i < len(tokens) - 1:
        if tokens[i] == first and tokens[i + 1] == second:
            result.append(merged)
            i += 2
        else:
            result.append(tokens[i])
            i += 1

    if i == len(tokens) - 1:
        result.append(tokens[i])

    return result


def _apply_merge_and_update_counts(
    tokens: list[int],
    first: int,
    second: int,
    merged: int,
    pair_counts: Counter[tuple[int, int]],
) -> list[int]:
    if len(tokens) < 2:
        return tokens

    result: list[int] = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == first and tokens[i + 1] == second:
            if result:
                prev = result[-1]
                pair_counts[(prev, first)] -= 1
                if pair_counts[(prev, first)] <= 0:
                    del pair_counts[(prev, first)]

            if i + 2 < len(tokens):
                next_token = tokens[i + 2]
                pair_counts[(second, next_token)] -= 1
                if pair_counts[(second, next_token)] <= 0:
                    del pair_counts[(second, next_token)]

            result.append(merged)

            if len(result) >= 2:
                prev = result[-2]
                pair_counts[(prev, merged)] += 1

            if i + 2 < len(tokens):
                next_token = tokens[i + 2]
                if not (i + 3 < len(tokens) and tokens[i + 2] == first and tokens[i + 3] == second):
                    pair_counts[(merged, next_token)] += 1

            i += 2
        else:
            result.append(tokens[i])
            i += 1

    return result


class BPETokenizer(BaseTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self._vocabulary: dict[int, bytes] = {}
        self._token_to_id: dict[bytes, int] = {}
        self._merge_rules: list[tuple[int, int, int]] = []
        self._special_tokens: dict[str, int] = {}

        self._init_special_tokens()
        self._init_byte_vocabulary()

    def _init_special_tokens(self) -> None:
        for i, token in enumerate(SpecialTokensEnum):
            token_bytes = token.encode("utf-8")
            self._vocabulary[i] = token_bytes
            self._token_to_id[token_bytes] = i
            self._special_tokens[token] = i

    def _init_byte_vocabulary(self) -> None:
        offset = len(self._special_tokens)
        for i in range(256):
            byte_val = bytes([i])
            token_id = offset + i
            self._vocabulary[token_id] = byte_val
            self._token_to_id[byte_val] = token_id

    @property
    def get_vocab_size(self) -> int:
        return len(self._vocabulary)

    @property
    def bos_id(self) -> int:
        return self._special_tokens[SpecialTokensEnum.BOS]

    @property
    def eos_id(self) -> int:
        return self._special_tokens[SpecialTokensEnum.EOS]

    @property
    def pad_id(self) -> int:
        return self._special_tokens[SpecialTokensEnum.PAD]

    def token_to_id(self, token: str | SpecialTokensEnum) -> int | None:
        return self._special_tokens.get(token)

    def id_to_token(self, token_id: int) -> bytes | None:
        return self._vocabulary.get(token_id)

    def encode(self, text: str) -> list[int]:
        byte_offset = len(self._special_tokens)
        tokens = [b + byte_offset for b in text.encode("utf-8")]

        for first, second, merged in self._merge_rules:
            tokens = _apply_merge_rule(tokens, first, second, merged)

        return tokens

    def encode_with_special_tokens(self, text: str) -> list[int]:
        return [self.bos_id, *self.encode(text), self.eos_id]

    def encode_batch(
        self,
        texts: list[str],
        add_special_tokens: bool = True,
    ) -> list[list[int]]:
        if add_special_tokens:
            return [self.encode_with_special_tokens(text) for text in texts]
        return [self.encode(text) for text in texts]

    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        special_ids = set(self._special_tokens.values()) if skip_special_tokens else set()

        byte_chunks = []
        for token in tokens:
            if token in special_ids or token not in self._vocabulary:
                continue
            byte_chunks.append(self._vocabulary[token])

        byte_sequence = b"".join(byte_chunks)
        return byte_sequence.decode("utf-8", errors="replace")

    @classmethod
    def train(cls, texts: list[str], **kwargs) -> Self:
        target_size: int = kwargs.get("target_size", 1000)
        verbose: bool = kwargs.get("verbose", False)
        log_every: int = kwargs.get("log_every", 100)

        tokenizer = cls()
        byte_offset = len(tokenizer._special_tokens)

        if verbose:
            print("Encoding texts to bytes...")

        encoded_texts: list[list[int]] = []
        for text in texts:
            encoded = [b + byte_offset for b in text.encode("utf-8")]
            encoded_texts.append(encoded)

        if verbose:
            total_tokens = sum(len(t) for t in encoded_texts)
            print(f"Total tokens: {total_tokens:,} across {len(texts):,} texts")
            print("Building initial pair counts...")

        pair_counts: Counter[tuple[int, int]] = Counter()
        for tokens in encoded_texts:
            pair_counts.update(zip(tokens[:-1], tokens[1:], strict=False))

        num_merges = target_size - len(tokenizer._vocabulary)
        start_time = time.time()
        last_log_time = start_time

        for step in range(num_merges):
            if not pair_counts:
                break

            best_pair = pair_counts.most_common(1)[0][0]
            first, second = best_pair

            new_token_id = len(tokenizer._vocabulary)
            new_token_bytes = tokenizer._vocabulary[first] + tokenizer._vocabulary[second]

            tokenizer._vocabulary[new_token_id] = new_token_bytes
            tokenizer._token_to_id[new_token_bytes] = new_token_id
            tokenizer._merge_rules.append((first, second, new_token_id))

            for i, tokens in enumerate(encoded_texts):
                encoded_texts[i] = _apply_merge_and_update_counts(tokens, first, second, new_token_id, pair_counts)

            del pair_counts[best_pair]

            if verbose and (step + 1) % log_every == 0:
                now = time.time()
                elapsed = now - start_time
                step_time = now - last_log_time
                steps_remaining = num_merges - (step + 1)
                eta = (elapsed / (step + 1)) * steps_remaining
                total_tokens = sum(len(t) for t in encoded_texts)

                print(
                    f"Step {step + 1:,}/{num_merges:,} "
                    f"({100 * (step + 1) / num_merges:.1f}%) | "
                    f"tokens: {total_tokens:,} | "
                    f"{log_every} steps in {step_time:.1f}s | "
                    f"ETA: {eta:.0f}s"
                )
                last_log_time = now

        if verbose:
            total_time = time.time() - start_time
            print(f"Training complete in {total_time:.1f}s. Final vocab size: {tokenizer.get_vocab_size}")

        return tokenizer

    def save(self, path: str | Path) -> None:
        path = Path(path)

        vocab_serializable = {str(k): v.hex() for k, v in self._vocabulary.items()}

        data = {
            "vocabulary": vocab_serializable,
            "merge_rules": self._merge_rules,
            "special_tokens": self._special_tokens,
        }

        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Self:
        path = Path(path)
        data = json.loads(path.read_text())

        tokenizer = cls.__new__(cls)
        tokenizer._vocabulary = {int(k): bytes.fromhex(v) for k, v in data["vocabulary"].items()}
        tokenizer._token_to_id = {v: k for k, v in tokenizer._vocabulary.items()}
        tokenizer._merge_rules = [(rule[0], rule[1], rule[2]) for rule in data["merge_rules"]]
        tokenizer._special_tokens = data["special_tokens"]

        return tokenizer
