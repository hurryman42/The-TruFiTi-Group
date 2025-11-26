import torch


def train_val_test_split(
    data: list[int],
    train_size: float = 0.9,
    val_size: float = 0.1,
    test_size: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if test_size is None:
        assert train_size + val_size == 1.0, f"train_size + val_size must equal 1.0, got {train_size + val_size}"
    else:
        assert (
            train_size + val_size + test_size == 1.0
        ), f"train_size + val_size + test_size must equal 1.0, got {train_size + val_size + test_size}"

    data_tensor = torch.tensor(data, dtype=torch.long)
    n = len(data_tensor)

    train_end = int(train_size * n)
    val_end = train_end + int(val_size * n)

    train_data = data_tensor[:train_end]
    val_data = data_tensor[train_end:val_end]
    test_data = data_tensor[val_end:] if test_size is not None else None

    return train_data, val_data, test_data


def get_batch(
    data: torch.Tensor,
    seq_len: int,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - seq_len, (batch_size,))

    x = torch.stack([data[i : i + seq_len] for i in ix])
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix])

    x, y = x.to(device), y.to(device)

    return x, y
