from pathlib import Path

import torch

from src.models.bigram_language_model import BigramLanguageModel
from src.models.embeddings.positional_encoding import PositionalEncoding
from src.models.embeddings.token_embedding import TokenEmbedding
from src.tokenizer.char_tokenizer import CharTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent

KEY_VOCAB_SIZE = "vocab_size"
KEY_D_MODEL = "d_model"
KEY_MODEL = "model"
KEY_TOKEN_EMBEDDING = "token_embedding"
KEY_POS_ENCODING = "pos_encoding"
KEY_SEQ_LEN = "seq_len"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}\n")

tokenizer = CharTokenizer.load(str(BASE_DIR.parent / "tokenizer" / "char_tokenizer.json"))
checkpoint = torch.load(BASE_DIR.parent / "models" / "bigram_model.pt", map_location=device)

VOCAB_SIZE = checkpoint[KEY_VOCAB_SIZE]
D_MODEL = checkpoint[KEY_D_MODEL]
SEQ_LEN = checkpoint[KEY_SEQ_LEN]

# Initialize models
token_embedding = TokenEmbedding(VOCAB_SIZE, D_MODEL, scale=False).to(device)
pos_encoding = PositionalEncoding(D_MODEL, max_seq_len=SEQ_LEN).to(device)
model = BigramLanguageModel(VOCAB_SIZE, D_MODEL).to(device)

# Load weights
token_embedding.load_state_dict(checkpoint[KEY_TOKEN_EMBEDDING])
pos_encoding.load_state_dict(checkpoint[KEY_POS_ENCODING])
model.load_state_dict(checkpoint[KEY_MODEL])

model.eval()


def generate(prompt="", length=200):
    if prompt:
        idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)

    result = model.generate(token_embedding, pos_encoding, idx, length, max_context_len=128)
    return tokenizer.decode(result[0].tolist())


print(generate(""))
print("\n" + "-" * 80 + "\n")
print(generate("The film "))
print("\n" + "-" * 80 + "\n")
print(generate("A story about "))
