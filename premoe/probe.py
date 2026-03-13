"""Linear probe for expert routing prediction."""

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from premoe.config import PreMoEConfig


class LinearProbe(nn.Module):
    """Lightweight linear probe: hidden_dim -> num_experts.

    Predicts expert routing from pre-attention hidden states.
    Probe weight shape: [num_experts, hidden_dim].
    """

    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def load_probes(
    config: PreMoEConfig,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[int, LinearProbe]:
    """Load pre-trained probe weights for each anchor layer.

    Expects files named `probe_layer{idx}.pt` in config.probe_dir,
    each containing a state dict with key "linear.weight" of shape
    [num_experts, hidden_dim].

    Returns:
        Dict mapping layer index to loaded LinearProbe on the target device.
    """
    probe_dir = Path(config.probe_dir)
    probes: Dict[int, LinearProbe] = {}

    for layer_idx in config.anchor_layers:
        path = probe_dir / f"probe_layer{layer_idx}.pt"
        if not path.exists():
            print(f"[Pre-MoE] WARNING: probe not found for layer {layer_idx}: {path}")
            continue

        probe = LinearProbe(config.hidden_dim, config.num_experts)
        state = torch.load(path, map_location=device, weights_only=True)
        probe.load_state_dict(state)
        probe = probe.to(device=device, dtype=dtype)
        probe.eval()
        for p in probe.parameters():
            p.requires_grad_(False)
        probes[layer_idx] = probe
        print(f"[Pre-MoE] Loaded probe for layer {layer_idx} "
              f"(shape={probe.linear.weight.shape})")

    return probes
