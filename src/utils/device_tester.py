import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    print("Using device:", device, "| GPU:", name)
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device:", device)
else:
    device = torch.device("cpu")
    print("Using device:", device)

print("ROCm available:", torch.version.hip is not None)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("Compiled for HIP architectures:", torch.cuda.get_arch_list())
