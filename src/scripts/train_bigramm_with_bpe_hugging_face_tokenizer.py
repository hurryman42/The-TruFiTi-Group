from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.models.bigram_language_model import BigramLanguageModel
from src.models.embeddings.positional_encoding import PositionalEncoding
from src.models.embeddings.token_embedding import TokenEmbedding
from src.scripts.read_file import read_file_only_reviews

BASE_DIR = Path(__file__).resolve().parent.parent
DIMENSION_MODEL = 256
SEQ_LEN = 128
BATCH_SIZE = 64
EVAL_ITERS = 50  # Number of batches to average for loss estimation
MAX_ITERS = 3000
EVAL_INTERVAL = 1000  # Interval of printing estimated loss
LEARNING_RATE = 1e-3


def load_bpe_tokenizer() -> Tokenizer:
    tokenizer_path = BASE_DIR.parent / "tokenizer" / "bpe_hugging_face_tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}\n")
    return tokenizer


def load_text() -> list:
    input_file = BASE_DIR.parent / "data" / "letterboxd_filtered_short_synopsis_film.jsonl"
    texts = read_file_only_reviews(input_file)

    print(f"Number of reviews: {len(texts):,}".replace(",", "."))
    return texts


def encode_texts_batched(texts: list[str], tokenizer: Tokenizer, batch_size: int = 1000) -> list[int]:
    all_ids = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded_batch = tokenizer.encode_batch(batch)

        for encoded in encoded_batch:
            all_ids.extend(encoded.ids)

        if i % 200000 == 0:
            print(f"Encoded {i:,} / {len(texts):,} texts...")

    print(f"Total tokens: {len(all_ids):,}")
    return all_ids


def train_val_split(data: list[int], train_size: float) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.tensor(data, dtype=torch.long)
    n = int(train_size * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data


def get_batch(
    split: str,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    seq_len: int,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == "train" else val_data

    ix = torch.randint(len(data) - seq_len, (batch_size,))  # (batch_size,)

    x = torch.stack([data[i : i + seq_len] for i in ix])
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix])

    x, y = x.to(device), y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss(
    model,
    token_embedding,
    pos_encoding,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    seq_len: int,
    batch_size: int,
    eval_iters: int,
    device: str,
) -> dict:
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, train_data, val_data, seq_len, batch_size, device)

            tok_emb = token_embedding(x)
            embeddings = pos_encoding(tok_emb)

            logits, loss = model(embeddings, y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out


def train(
    model: BigramLanguageModel,
    token_embedding: TokenEmbedding,
    pos_encoding: PositionalEncoding,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    optimizer,
    max_iters: int,
    device: str,
):
    print("Starting training...\n")

    for iter in range(max_iters):
        if iter % EVAL_INTERVAL == 0 or iter == max_iters - 1:
            losses = estimate_loss(
                model,
                token_embedding,
                pos_encoding,
                train_data,
                val_data,
                SEQ_LEN,
                BATCH_SIZE,
                EVAL_ITERS,
                device,
            )
            print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch("train", train_data, val_data, SEQ_LEN, BATCH_SIZE, device)

        tok_emb = token_embedding(xb)
        embeddings = pos_encoding(tok_emb)

        logits, loss = model(embeddings, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("\nTraining completed!")


def save_model(
    model: BigramLanguageModel,
    token_embedding: TokenEmbedding,
    pos_encoding: PositionalEncoding,
    vocab_size: int,
):
    save_path = BASE_DIR.parent / "models" / "bigram_model_bpe_with_hugging_face.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "token_embedding": token_embedding.state_dict(),
            "pos_encoding": pos_encoding.state_dict(),
            "vocab_size": vocab_size,
            "dimension_model": DIMENSION_MODEL,
            "seq_len": SEQ_LEN,
        },
        save_path,
    )

    print(f"\nModel saved to {save_path}")


def main():
    # Setup
    tokenizer = load_bpe_tokenizer()

    vocab_size = tokenizer.get_vocab_size()
    texts = load_text()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Prepare data
    encoded_texts = encode_texts_batched(texts, tokenizer)
    train_data, val_data = train_val_split(encoded_texts, 0.9)

    # Initialize models
    token_embedding = TokenEmbedding(vocab_size, DIMENSION_MODEL, scale=False).to(device)
    pos_encoding = PositionalEncoding(DIMENSION_MODEL, max_seq_len=SEQ_LEN).to(device)
    model = BigramLanguageModel(vocab_size, DIMENSION_MODEL).to(device)
    print("Initialize models finished")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}\n".replace(",", "."))

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(token_embedding.parameters()),
        lr=LEARNING_RATE,
    )

    train(
        model,
        token_embedding,
        pos_encoding,
        train_data,
        val_data,
        optimizer,
        MAX_ITERS,
        device,
    )

    save_model(model, token_embedding, pos_encoding, vocab_size)


if __name__ == "__main__":
    main()
