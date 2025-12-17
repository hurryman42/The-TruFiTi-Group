import torch

from src.utils.device import get_device

device = get_device()

if device == "cuda":
    name = torch.cuda.get_device_name(0)
    print("Using device:", device, "| GPU:", name)
    print("ROCm available:", torch.version.hip is not None)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("Compiled for HIP architectures:", torch.cuda.get_arch_list())
else:
    print("Using device:", device)
