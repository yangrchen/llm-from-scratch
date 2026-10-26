import re
from pathlib import Path

from llm_from_scratch.tokenize.simple import SimpleTokenizerV1

ROOT = Path(__file__).resolve().parents[3]
DATA_FOLDER = ROOT / "data"

if __name__ == "__main__":
    # Exploring the text excerpt that we are using as pretraining data
    with open(DATA_FOLDER / "the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    # print("Total number of characters:", len(raw_text))
    # print(raw_text[:99])

    # Exploring splitting
    text = "Hello, world. This, is a test."
    result = re.split(r"(\s)", text)
    # print(result)

    result = re.split(r"([,.]|\s)", text)
    # print(result)

    text = "Hello, world. Is this-- a test?"
    result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    result = [item.strip() for item in result if item.strip()]
    # print(result)

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    # print("Length of `preprocessed`:", len(preprocessed))

    # print(preprocessed[:30])

    # Converting tokens into integer representations to produce token IDs
    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words)
    # print("Vocab size:", vocab_size)

    vocab = {token: integer for integer, token in enumerate(all_words)}
    for i, item in enumerate(vocab.items()):
        # print(item)
        if i >= 50:
            break

    # Using the simple tokenizer class
    tokenizer = SimpleTokenizerV1(vocab)
    text = """"It's the last he painted, you know,"
            Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    print(ids)


