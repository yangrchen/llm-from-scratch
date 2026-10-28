import itertools
import re

from transformers import AutoTokenizer

_tokenizer = AutoTokenizer.from_pretrained("gpt2")


class SimpleTokenizerV1:
    def __init__(self, vocab: dict[str, int]) -> None:
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: list[int]) -> str:
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text


class SimpleTokenizerV2:
    def __init__(self, vocab: dict[str, int]) -> None:
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: list[int]) -> str:
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r"\1", text)
        return text


# --- Karpathy MinBPE


def get_stats(
    ids: list[int], counts: dict[tuple[int, int], int] | None = None
) -> dict[tuple[int, int], int]:
    """Given a list of integers, return a dictionary of counts of consecutive pairs."""
    counts = {} if counts is None else counts
    for pair in itertools.pairwise(ids):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], idx: int):
    """Replace all consecutive occurrences of a pair with the new token ``idx``."""
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class Tokenizer:
    """Base class for tokenizers."""

    def __init__(self) -> None:
        self.merges = {}
        self.pattern = ""
        self.special_tokens = {}
        self.vocab = self._build_vocab()

    def train(self, text: str, vocab_size: int, verbose=False):
        raise NotImplementedError

    def encode(self, text: str):
        raise NotImplementedError

    def decode(self, ids):
        raise NotImplementedError

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab


class BasicBPE(Tokenizer):
    def __init__(self) -> None:
        super().__init__()

    def train(self, text: str, vocab_size: int, verbose=False) -> None:
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        # Iteratively merge most common pairs to create new tokens up to `vocab_size`
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            # Count number of times every consecutive pair appears to find most common pair
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)

            # Assign next available ID
            idx = 256 + i
            ids = merge(ids, pair, idx)

            # Save the merge and the new vocab
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(
                    f"Merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences"
                )

            self.merges = merges
            self.vocab = vocab

    def decode(self, ids: list[int]) -> str:
        text = b"".join(self.vocab[idx] for idx in ids)
        return text.decode("utf-8", errors="replace")

    def encode(self, text: str) -> list[int]:
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
