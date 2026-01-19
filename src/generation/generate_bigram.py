import torch
import argparse
from pathlib import Path

from src.dto.config import Config
from src.enums import BigramCheckpointEnum, TokenizerTypeEnum
from src.models.bigram_language_model import BigramLanguageModel
from src.models.embeddings.positional_encoding import PositionalEncoding
from src.models.embeddings.token_embedding import TokenEmbedding
from src.utils.device import get_device

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def load_model(model_path: Path):
    device = get_device()
    print(f"Using device: {device}")

    checkpoint = torch.load(model_path, map_location=device)

    config = Config.from_dict(checkpoint[BigramCheckpointEnum.CONFIG])
    tokenizer = checkpoint[BigramCheckpointEnum.TOKENIZER]
    vocab_size = checkpoint[BigramCheckpointEnum.VOCAB_SIZE]

    print(f"Tokenizer: {config.tokenizer.type}")
    print(f"Vocab size: {vocab_size}, d_model: {config.model.d_model}, seq_len: {config.model.seq_len}\n")

    token_embedding = TokenEmbedding(vocab_size, config.model.d_model, scale=False).to(device)
    pos_encoding = PositionalEncoding(config.model.d_model, max_seq_len=config.model.seq_len).to(device)
    model = BigramLanguageModel(vocab_size, config.model.d_model).to(device)

    token_embedding.load_state_dict(checkpoint[BigramCheckpointEnum.TOKEN_EMBEDDING])
    pos_encoding.load_state_dict(checkpoint[BigramCheckpointEnum.POS_ENCODING])
    model.load_state_dict(checkpoint[BigramCheckpointEnum.MODEL])

    model.eval()

    return model, token_embedding, pos_encoding, tokenizer, config, device


def generate(
    model,
    token_embedding,
    pos_encoding,
    tokenizer,
    config: Config,
    device,
    prompt: str = "",
    length: int = 200,
) -> str:
    if prompt:
        if config.tokenizer.type == TokenizerTypeEnum.CHAR:
            idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        else:
            encoded = tokenizer.encode(prompt)
            idx = torch.tensor(encoded.ids, dtype=torch.long, device=device).unsqueeze(0)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)

    result = model.generate(token_embedding, pos_encoding, idx, length, max_context_len=config.model.seq_len)

    return tokenizer.decode(result[0].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with Bigram Language Model")
    parser.add_argument("--model", type=str, default="bigram_model_bpe_hugging_face.pt", help="Model filename")
    parser.add_argument("--prompt", type=str, default="", help="Prompt for generation")
    parser.add_argument("--length", type=int, default=200, help="Number of tokens to generate")

    args = parser.parse_args()

    model_path = BASE_DIR / "models" / args.model
    model, token_embedding, pos_encoding, tokenizer, config, device = load_model(model_path)

    print("=" * 80)
    print(f"Prompt: '{args.prompt}'" if args.prompt else "Unconditional generation:")
    print("=" * 80)
    print(generate(model, token_embedding, pos_encoding, tokenizer, config, device, args.prompt, args.length))
