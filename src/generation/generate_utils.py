from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.config.config import Config, BigramModelConfig, TransformerModelConfig, GRUModelConfig
from src.enums import TransformerCheckpointEnum, BigramCheckpointEnum, GruCheckpointEnum
from src.enums.types import SpecialTokensEnum, ModelTypeEnum
from src.models.bigram.bigram import Bigram
from src.models.gru.gru import GRU
from src.models.transformer.transformer import TransformerDecoderOnly


def decode_generated(
    generated: torch.Tensor,
    tokenizer: Tokenizer,
    eos_id: int,
) -> list[str]:
    results = []
    for tokens in generated.tolist():
        if eos_id in tokens:
            tokens = tokens[: tokens.index(eos_id)]
        results.append(tokenizer.decode(tokens))
    return results


def print_generation_header(prompt: str):
    print("=" * 80)
    if prompt:
        print(f"Prompt: '{prompt}'")
    else:
        print(f"Unconditional generation (starting with {SpecialTokensEnum.BOS}):")
    print("=" * 80)


def print_results(results: list[str], num_generations: int):
    if num_generations == 1:
        print(results[0])
    else:
        for i, text in enumerate(results, 1):
            print(f"\n--- Generation {i} ---")
            print(text)


def load_model_checkpoint(checkpoint_path: Path, device: torch.device, model_type):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    match model_type:
        case ModelTypeEnum.BIGRAM:
            config = Config.from_dict(checkpoint[BigramCheckpointEnum.CONFIG])
            assert isinstance(config.model, BigramModelConfig)
            bigram_model_config: BigramModelConfig = config.model

            tokenizer = checkpoint[BigramCheckpointEnum.TOKENIZER]
            vocab_size = checkpoint[BigramCheckpointEnum.VOCAB_SIZE]

            print(f"Tokenizer: {config.tokenizer.type}")
            print(f"Vocab size: {vocab_size}, d_model: {bigram_model_config.d_model}")

            bigram = Bigram(vocab_size, bigram_model_config.d_model).to(device)

            bigram.load_state_dict(checkpoint[BigramCheckpointEnum.MODEL])
            bigram.eval()

            return bigram, tokenizer, config

        case ModelTypeEnum.GRU:
            config = Config.from_dict(checkpoint[str(GruCheckpointEnum.CONFIG)])
            assert isinstance(config.model, GRUModelConfig)
            gru_model_config: GRUModelConfig = config.model

            tokenizer = checkpoint[str(GruCheckpointEnum.TOKENIZER)]
            vocab_size = checkpoint[str(GruCheckpointEnum.VOCAB_SIZE)]

            print(f"Tokenizer: {config.tokenizer.name}")
            print(
                f"Vocab size: {vocab_size}, input_size: {gru_model_config.input_size},"
                f" hidden_size: {gru_model_config.hidden_size}\n"
            )

            gru = GRU(
                vocab_size=vocab_size,
                input_size=gru_model_config.input_size,
                hidden_size=gru_model_config.hidden_size,
                num_layers=gru_model_config.num_layers,
                dropout=gru_model_config.dropout,
            ).to(device)

            gru.load_state_dict(checkpoint[str(GruCheckpointEnum.MODEL)])
            gru.eval()

            return gru, tokenizer, config

        case ModelTypeEnum.TRANSFORMER:
            config = Config.from_dict(checkpoint[str(TransformerCheckpointEnum.CONFIG)])
            assert isinstance(config.model, TransformerModelConfig)
            transformer_model_config: TransformerModelConfig = config.model

            tokenizer = checkpoint[str(TransformerCheckpointEnum.TOKENIZER)]
            vocab_size = checkpoint[str(TransformerCheckpointEnum.VOCAB_SIZE)]

            print(f"Tokenizer: {config.tokenizer.name}")
            print(
                f"Vocab size: {vocab_size}, d_model: {transformer_model_config.d_model},"
                f" seq_len: {transformer_model_config.seq_len}"
            )
            print(f"Blocks: {transformer_model_config.num_blocks}, Heads: {transformer_model_config.num_heads}\n")

            transformer = TransformerDecoderOnly(
                vocab_size,
                embedding_dimension=transformer_model_config.d_model,
                num_blocks=transformer_model_config.num_blocks,
                num_heads=transformer_model_config.num_heads,
                head_dimension=transformer_model_config.head_dim,
                max_seq_len=transformer_model_config.seq_len,
                ff_hidden_dimension=transformer_model_config.ff_hidden_dim,
                dropout=transformer_model_config.dropout,
                use_rope=transformer_model_config.use_rope,
            ).to(device)

            transformer.load_state_dict(checkpoint[str(TransformerCheckpointEnum.MODEL)])
            transformer.eval()

            return transformer, tokenizer, config

        case _:
            raise ValueError(f"Unknown model type: {model_type}")
