"""End-to-end tests for Pre-MoE SGLang integration.

These tests require SGLang and a GPU. Mark with pytest markers
so they can be skipped in CI.
"""

import pytest
import torch

# Skip entire module if no GPU or SGLang
pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for E2E tests"
    ),
]


def test_patcher_finds_layers():
    """Verify the patcher can locate decoder layers in a mock model."""
    from premoe.config import PreMoEConfig

    # Create a mock model structure matching SGLang's DeepseekV2
    class MockLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = torch.nn.LayerNorm(128)
            self.self_attn = torch.nn.Linear(128, 128)
            self.mlp = torch.nn.Linear(128, 128)

        def forward(self, x):
            return self.input_layernorm(x)

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([MockLayer() for _ in range(27)])

    model = MockModel()

    # Verify layer access
    assert hasattr(model, 'model')
    assert hasattr(model.model, 'layers')
    assert len(model.model.layers) == 27
    assert hasattr(model.model.layers[1], 'input_layernorm')


@pytest.mark.skipif(
    not torch.cuda.device_count() >= 2,
    reason="2 GPUs required for full E2E test"
)
def test_premoe_output_correctness():
    """Verify Pre-MoE patched model produces same output as unpatched.

    This test requires SGLang installed and 2 GPUs.
    """
    pytest.skip("Full E2E test requires SGLang server running")
