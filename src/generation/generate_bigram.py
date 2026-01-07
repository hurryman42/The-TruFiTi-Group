import torch
import argparse
from pathlib import Path

from src.enums import CheckpointEnum, TokenizerTypeEnum
from src.models.bigram_language_model import BigramLanguageModel
from src.models.embeddings.positional_encoding import PositionalEncoding
from src.models.embeddings.token_embedding import TokenEmbedding
from src.utils.device import get_device
from src.utils.tokenizer_loader import load_tokenizer

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def load_model(model_path: Path):
    device = get_device()
    print(f"Using device: {device}")

    checkpoint = torch.load(model_path, map_location=device)

    tokenizer_type = TokenizerTypeEnum(checkpoint[CheckpointEnum.TOKENIZER_TYPE])
    tokenizer_name = checkpoint[CheckpointEnum.TOKENIZER_NAME]
    tokenizer_path = BASE_DIR / "tokenizer" / tokenizer_name

    print(f"Tokenizer: {tokenizer_type}")
    print(
        f"Vocab size: {checkpoint[CheckpointEnum.VOCAB_SIZE]}, "
        f"d_model: {checkpoint[CheckpointEnum.D_MODEL]}, "
        f"seq_len: {checkpoint[CheckpointEnum.SEQ_LEN]}\n"
    )

    tokenizer = load_tokenizer(tokenizer_type, tokenizer_path)

    token_embedding = TokenEmbedding(
        checkpoint[CheckpointEnum.VOCAB_SIZE], checkpoint[CheckpointEnum.D_MODEL], scale=False
    ).to(device)
    pos_encoding = PositionalEncoding(
        checkpoint[CheckpointEnum.D_MODEL], max_seq_len=checkpoint[CheckpointEnum.SEQ_LEN]
    ).to(device)
    model = BigramLanguageModel(checkpoint[CheckpointEnum.VOCAB_SIZE], checkpoint[CheckpointEnum.D_MODEL]).to(device)

    token_embedding.load_state_dict(checkpoint[CheckpointEnum.TOKEN_EMBEDDING])
    pos_encoding.load_state_dict(checkpoint[CheckpointEnum.POS_ENCODING])
    model.load_state_dict(checkpoint[CheckpointEnum.MODEL])

    model.eval()

    return model, token_embedding, pos_encoding, tokenizer, tokenizer_type, checkpoint[CheckpointEnum.SEQ_LEN], device


def generate(
    model,
    token_embedding,
    pos_encoding,
    tokenizer,
    tokenizer_type,
    seq_len,
    device,
    prompt: str = "",
    length: int = 200,
) -> str:
    if prompt:
        if tokenizer_type == TokenizerTypeEnum.CHAR:
            idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        else:
            encoded = tokenizer.encode(prompt)
            idx = torch.tensor(encoded.ids, dtype=torch.long, device=device).unsqueeze(0)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)

    result = model.generate(token_embedding, pos_encoding, idx, length, max_context_len=seq_len)

    if tokenizer_type == TokenizerTypeEnum.CHAR:
        return tokenizer.decode(result[0].tolist())
    return tokenizer.decode(result[0].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with Bigram Language Model")
    parser.add_argument("--model", type=str, default="bigram_model_bpe_hugging_face.pt", help="Model filename")
    parser.add_argument("--prompt", type=str, default="", help="Prompt for generation")
    parser.add_argument("--length", type=int, default=200, help="Number of tokens to generate")

    args = parser.parse_args()

    model_path = BASE_DIR / "models" / args.model
    model, token_embedding, pos_encoding, tokenizer, tokenizer_type, seq_len, device = load_model(model_path)

    print("=" * 80)
    print(f"Prompt: '{args.prompt}'" if args.prompt else "Unconditional generation:")
    print("=" * 80)
    print(
        generate(
            model, token_embedding, pos_encoding, tokenizer, tokenizer_type, seq_len, device, args.prompt, args.length
        )
    )
