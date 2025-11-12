import torch

from src.models.embeddings.token_embedding import TokenEmbedding


def test_initialization():
    vocab_size = 100
    dim_embedding = 512

    token_emb = TokenEmbedding(vocab_size, dim_embedding)

    assert isinstance(token_emb.embedding, torch.nn.Embedding)

    assert token_emb.embedding.num_embeddings == vocab_size
    assert token_emb.embedding.embedding_dim == dim_embedding
    assert token_emb.dim_embedding == dim_embedding


def test_forward_shape():
    vocab_size = 50
    dim_embedding = 128
    batch_size = 4
    seq_len = 10

    token_emb = TokenEmbedding(vocab_size, dim_embedding)

    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

    output = token_emb(tokens)

    assert output.shape == (batch_size, seq_len, dim_embedding)