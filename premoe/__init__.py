"""Pre-MoE: Speculative Expert Pre-Dispatch for SGLang."""

__version__ = "0.1.0"

from premoe.config import PreMoEConfig
from premoe.probe import LinearProbe, load_probes
from premoe.patcher import patch_sglang_for_premoe
