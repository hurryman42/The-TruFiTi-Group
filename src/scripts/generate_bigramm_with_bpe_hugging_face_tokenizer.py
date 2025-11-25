from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.models.bigram_language_model import BigramLanguageModel
from src.models.embeddings.positional_encoding import PositionalEncoding
from src.models.embeddings.token_embedding import TokenEmbedding

BASE_DIR = Path(__file__).resolve().parent.parent

KEY_VOCAB_SIZE = "vocab_size"
KEY_DIMENSION_MODEL = "dimension_model"
KEY_MODEL = "model"
KEY_TOKEN_EMBEDDING = "token_embedding"
KEY_POS_ENCODING = "pos_encoding"
KEY_SEQ_LEN = "seq_len"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}\n")

tokenizer = Tokenizer.from_file(str(BASE_DIR.parent / "tokenizer" / "bpe_hugging_face_tokenizer.json"))
print(f"Tokenizer vocabulary size: {tokenizer.get_vocab_size()}")

checkpoint = torch.load(BASE_DIR.parent / "models" / "bigram_model_bpe_with_hugging_face.pt", map_location=device)

VOCAB_SIZE = checkpoint[KEY_VOCAB_SIZE]
DIMENSION_MODEL = checkpoint[KEY_DIMENSION_MODEL]
SEQ_LEN = checkpoint[KEY_SEQ_LEN]

print(f"Model vocabulary size: {VOCAB_SIZE}")
print(f"Model dimension: {DIMENSION_MODEL}")
print(f"Sequence length: {SEQ_LEN}\n")

# Initialize models
token_embedding = TokenEmbedding(VOCAB_SIZE, DIMENSION_MODEL, scale=False).to(device)
pos_encoding = PositionalEncoding(DIMENSION_MODEL, max_seq_len=SEQ_LEN).to(device)
model = BigramLanguageModel(VOCAB_SIZE, DIMENSION_MODEL).to(device)

# Load weights
token_embedding.load_state_dict(checkpoint[KEY_TOKEN_EMBEDDING])
pos_encoding.load_state_dict(checkpoint[KEY_POS_ENCODING])
model.load_state_dict(checkpoint[KEY_MODEL])

model.eval()


def generate(prompt: str = "", length: int = 200) -> str:
    if prompt:
        encoded = tokenizer.encode(prompt)
        idx = torch.tensor(encoded.ids, dtype=torch.long, device=device).unsqueeze(0)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    result = model.generate(token_embedding, pos_encoding, idx, length, max_context_len=SEQ_LEN)

    generated_ids = result[0].tolist()
    return tokenizer.decode(generated_ids)


if __name__ == "__main__":
    print("=" * 80)
    print("Unconditional generation:")
    print("=" * 80)
    print(generate(""))

    print("\n" + "-" * 80 + "\n")

    print("=" * 80)
    print("Prompt: 'The film '")
    print("=" * 80)
    print(generate("The film "))

    print("\n" + "-" * 80 + "\n")

    print("=" * 80)
    print("Prompt: 'A story about '")
    print("=" * 80)
    print(generate("A story about "))
