"""Launch script: Start SGLang with Pre-MoE patches applied.

Usage:
    # Start SGLang server with Pre-MoE on 2 GPUs
    python scripts/run_sglang_premoe.py \
        --model deepseek-ai/DeepSeek-V2-Lite-Chat \
        --probe-dir probes \
        --tp 2

    # With custom anchor layers and profiling
    python scripts/run_sglang_premoe.py \
        --model deepseek-ai/DeepSeek-V2-Lite-Chat \
        --probe-dir probes \
        --anchor-layers 1 14 26 \
        --log-accuracy \
        --tp 2
"""

import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(
        description="Launch SGLang with Pre-MoE speculative dispatch"
    )
    parser.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-V2-Lite-Chat",
        help="Model name or path",
    )
    parser.add_argument(
        "--probe-dir",
        default="probes",
        help="Directory containing trained probe weights",
    )
    parser.add_argument(
        "--anchor-layers",
        type=int,
        nargs="+",
        default=[1, 14, 26],
        help="Layer indices to apply Pre-MoE",
    )
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallelism degree")
    parser.add_argument(
        "--ep", type=int, default=2, help="Expert parallelism degree"
    )
    parser.add_argument(
        "--log-accuracy",
        action="store_true",
        help="Log per-layer dispatch accuracy",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable verification (maximum overlap, may affect quality)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()

    # Import after parsing to fail fast on bad args
    try:
        import sglang as sgl
    except ImportError:
        print("ERROR: sglang not installed. Install with: pip install sglang")
        sys.exit(1)

    from premoe.config import PreMoEConfig
    from premoe.patcher import patch_sglang_for_premoe

    # Build Pre-MoE config
    config = PreMoEConfig(
        probe_dir=args.probe_dir,
        anchor_layers=args.anchor_layers,
        ep_size=args.ep,
        enable_verification=not args.no_verify,
        enable_fallback=not args.no_verify,
        log_accuracy=args.log_accuracy,
    )

    print("=" * 60)
    print("  Pre-MoE × SGLang PoC")
    print("=" * 60)
    print(f"  Model:         {args.model}")
    print(f"  Anchor layers: {config.anchor_layers}")
    print(f"  EP size:       {config.ep_size}")
    print(f"  Verification:  {config.enable_verification}")
    print(f"  Probe dir:     {config.probe_dir}")
    print("=" * 60)

    # SGLang server launch with Pre-MoE hook
    # The patching happens after the model is loaded but before serving starts.
    #
    # SGLang's launch flow:
    #   sgl.launch_server() -> loads model -> starts serving
    #
    # For the PoC, we use SGLang's model loading then apply patches.
    # This requires hooking into SGLang's initialization.

    # Method 1: Use SGLang's offline engine for direct model access
    print("\n[Pre-MoE] Loading model via SGLang Runtime...")

    runtime = sgl.Runtime(
        model_path=args.model,
        tp_size=args.tp,
    )

    # Access the model workers and apply patches
    # Note: SGLang's model is distributed across TP workers.
    # For EP=2, each worker holds a subset of experts.
    # The patcher needs to run on each worker.
    rank = int(os.environ.get("LOCAL_RANK", 0))

    # Get the underlying model from SGLang's runtime
    # This is SGLang-version-dependent
    model = _get_model_from_runtime(runtime)

    if model is not None:
        pipelines = patch_sglang_for_premoe(
            model, config, rank=rank, world_size=args.ep
        )
        print(f"[Pre-MoE] Patched {len(pipelines)} layers successfully")
    else:
        print("[Pre-MoE] WARNING: Could not access model from runtime. "
              "Falling back to unpatched mode.")

    # Start serving
    print(f"\n[Pre-MoE] Starting server on {args.host}:{args.port}")
    runtime.endpoint = f"http://{args.host}:{args.port}"

    # Keep the server running
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Pre-MoE] Shutting down...")
        # Print accuracy stats
        if model is not None:
            for p in pipelines:
                print(f"  Layer {p.layer_idx}: "
                      f"dispatch accuracy = {p.dispatch_accuracy:.4f} "
                      f"({p._total_tokens} tokens)")
        runtime.shutdown()


def _get_model_from_runtime(runtime):
    """Extract the underlying model from SGLang's Runtime.

    This is version-dependent and may need adjustment.
    SGLang's internal structure:
      Runtime -> TokenizerManager -> ModelRunner -> model
    """
    # Try common paths
    for attr_path in [
        "model",
        "model_runner.model",
        "engine.model_runner.model",
    ]:
        obj = runtime
        try:
            for attr in attr_path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue

    print("[Pre-MoE] Could not find model in runtime. "
          "Tried: model, model_runner.model, engine.model_runner.model")
    return None


if __name__ == "__main__":
    main()
