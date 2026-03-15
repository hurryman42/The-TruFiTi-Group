# ruff: noqa

import json
import matplotlib.pyplot as plt
import os

metric_files = [
    "transformer_sinusoidal_ffn4x_no-init_L2_0.7M_26-03-12_14-00-41_metrics.json",
    "transformer_sinusoidal_ffn4x_w-init_L2_0.7M_26-03-12_13-27-54_metrics.json",
    "transformer_rope_ffn4x_no-init_L2_0.7M_26-01-24_15-10-31_metrics.json",
    "transformer_rope_ffn4x_w-init_L2_0.7M_26-03-12_13-44-27_metrics.json",
]

labels = {
    "transformer_sinusoidal_ffn4x_no-init_L2_0.7M_26-03-12_14-00-41_metrics.json": "Transformer (Sinusoidal PE, Default Init)",
    "transformer_sinusoidal_ffn4x_w-init_L2_0.7M_26-03-12_13-27-54_metrics.json": "Transformer (Sinusoidal PE, Custom Init)",
    "transformer_rope_ffn4x_no-init_L2_0.7M_26-01-24_15-10-31_metrics.json": "Transformer (RoPE, Default Init)",
    "transformer_rope_ffn4x_w-init_L2_0.7M_26-03-12_13-44-27_metrics.json": "Transformer (RoPE, Custom Init)",
}


def load_metrics(path):
    with open(path) as f:
        data = json.load(f)
    return data["steps"], data["train_loss"], data["val_loss"]


global_min = min(min(load_metrics(f)[1] + load_metrics(f)[2]) for f in metric_files)
global_max = max(max(load_metrics(f)[1] + load_metrics(f)[2]) for f in metric_files)

fig, ax_val = plt.subplots(1, 1, figsize=(10, 5))

for file in metric_files:
    steps, train_loss, val_loss = load_metrics(file)
    label = labels.get(file, os.path.basename(file).replace("_metrics.json", ""))

    ax_val.plot(steps, val_loss, label=label)

ax_val.set_xlim(left=0, right=25000)
ax_val.minorticks_on()
ax_val.grid(which="both", axis="y", linestyle="--", color="gray", alpha=0.3)
ax_val.set_title("Validation Loss: Transformer Sinusoidal vs. RoPE")
ax_val.set_xlabel("Steps")
ax_val.set_ylabel("Loss")
ax_val.legend()

plt.tight_layout()
plt.savefig("init-compare.pdf")
plt.close()
