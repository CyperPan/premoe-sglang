"""Utilities for Pre-MoE: CUDA timing, logging."""

import torch


class CudaTimer:
    """Precise CUDA event-based timer for phase-level profiling."""

    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        self.start_event.record()

    def stop(self):
        self.end_event.record()

    def elapsed_ms(self) -> float:
        torch.cuda.synchronize()
        return self.start_event.elapsed_time(self.end_event)


class PhaseProfiler:
    """Collects per-phase timing across iterations for a single layer."""

    def __init__(self, layer_idx: int, enabled: bool = True):
        self.layer_idx = layer_idx
        self.enabled = enabled
        self.records: list[dict] = []

    def record(self, timings: dict):
        if self.enabled:
            self.records.append(timings)

    def summary(self) -> dict:
        if not self.records:
            return {}
        keys = self.records[0].keys()
        result = {}
        for k in keys:
            vals = [r[k] for r in self.records if isinstance(r.get(k), (int, float))]
            if vals:
                vals.sort()
                result[f"{k}_mean"] = sum(vals) / len(vals)
                result[f"{k}_p50"] = vals[len(vals) // 2]
        return result
