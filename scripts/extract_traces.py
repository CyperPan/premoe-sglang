"""Step 1: Extract real h_pre_attn traces from DeepSeek-V2-Lite.

Runs the model on long prompts and saves per-layer hidden states
and true gate routing decisions. These are used to:
  - Train real linear probes (Step 2)
  - Drive the closed-loop benchmark with real data (Step 3)

Usage:
    python scripts/extract_traces.py [--num-prompts 20] [--max-len 4096]
"""
import os
import json
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path


def get_long_prompts(num_prompts: int, target_tokens: int) -> list[str]:
    """Generate long English prompts by repeating diverse text."""
    base_texts = [
        "The development of large language models has transformed natural language processing. "
        "These models use transformer architectures with attention mechanisms that allow them to "
        "capture long-range dependencies in text. The scaling laws suggest that larger models "
        "trained on more data consistently improve performance across benchmarks. ",

        "In distributed computing systems, the challenge of minimizing communication overhead "
        "while maximizing computational throughput remains central. Expert parallelism divides "
        "model parameters across devices, requiring all-to-all communication patterns that can "
        "become bottlenecks at scale. Network topology and bandwidth constraints determine the "
        "practical limits of scaling. ",

        "Mixture of Experts models achieve computational efficiency by activating only a subset "
        "of parameters for each input token. The routing mechanism, typically a learned gate "
        "network, determines which experts process each token. Load balancing across experts "
        "is critical for training stability and inference efficiency. The sparse activation "
        "pattern creates unique challenges for distributed deployment. ",

        "Modern GPU architectures provide massive parallel computation through thousands of "
        "streaming multiprocessors. The memory hierarchy includes registers, shared memory, "
        "L2 cache, and high-bandwidth memory. Optimizing kernel launches and minimizing "
        "synchronization points are essential for achieving peak hardware utilization. ",

        "The attention mechanism computes a weighted sum over value vectors, where weights "
        "are determined by the compatibility between query and key vectors. Flash attention "
        "reduces memory footprint by computing attention in blocks, avoiding materialization "
        "of the full attention matrix. This enables processing of much longer sequences. ",
    ]

    prompts = []
    words_per_token = 1.3
    target_words = int(target_tokens * words_per_token)

    for i in range(num_prompts):
        base = base_texts[i % len(base_texts)]
        words_in_base = len(base.split())
        repeats = max(1, target_words // words_in_base + 1)
        prompt = (base * repeats)[:target_words * 6]
        prompts.append(prompt)

    return prompts


def extract_traces(args):
    """Extract hidden states and gate outputs from the model."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    # Identify MoE layer indices and gate weights
    moe_layers = []
    gate_weights = {}
    for name, param in model.named_parameters():
        if name.endswith(".mlp.gate.weight"):
            parts = name.split(".")
            layer_idx = int(parts[2])
            num_experts = param.shape[0]
            if num_experts > 512:
                continue
            moe_layers.append(layer_idx)
            gate_weights[layer_idx] = param.detach().cpu()
            print(f"  Found MoE gate: {name}, shape={param.shape}")

    moe_layers = sorted(set(moe_layers))
    print(f"Found {len(moe_layers)} MoE layers: {moe_layers[:5]}...{moe_layers[-3:]}")

    # Select anchor layers: early, mid, late
    if len(moe_layers) >= 3:
        anchor_layers = [moe_layers[0], moe_layers[len(moe_layers)//2], moe_layers[-1]]
    else:
        anchor_layers = moe_layers[:3]
    print(f"Anchor layers for probing: {anchor_layers}")

    for layer_idx in anchor_layers:
        torch.save(gate_weights[layer_idx], save_dir / f"gate_weight_layer{layer_idx}.pt")

    # Register hooks
    captured = {}
    hooks = []

    def make_pre_attn_hook(layer_idx):
        def hook_fn(module, args, kwargs=None):
            if isinstance(args, tuple) and len(args) > 0:
                h = args[0]
            else:
                return
            captured.setdefault(layer_idx, []).append(h.detach().cpu().to(torch.float16))
        return hook_fn

    def make_post_attn_hook(layer_idx):
        def hook_fn(module, args, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            key = f"{layer_idx}_post"
            captured.setdefault(key, []).append(h.detach().cpu().to(torch.float16))
        return hook_fn

    for layer_idx in anchor_layers:
        layer = model.model.layers[layer_idx]
        hooks.append(layer.input_layernorm.register_forward_hook(
            make_pre_attn_hook(layer_idx)
        ))
        hooks.append(layer.self_attn.register_forward_hook(
            make_post_attn_hook(layer_idx)
        ))

    # Generate prompts and run inference
    prompts = get_long_prompts(args.num_prompts, args.max_len)
    print(f"Running {len(prompts)} prompts (target ~{args.max_len} tokens each)...")

    all_traces = {layer_idx: {"h_pre": [], "h_post": [], "true_gate_ids": []}
                  for layer_idx in anchor_layers}

    for pi, prompt in enumerate(prompts):
        captured.clear()

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=args.max_len,
            truncation=True,
        ).to(model.device)

        actual_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            _ = model(**inputs, use_cache=False)

        for layer_idx in anchor_layers:
            if layer_idx in captured:
                h_pre = captured[layer_idx][0].squeeze(0)

                post_key = f"{layer_idx}_post"
                if post_key in captured:
                    h_post = captured[post_key][0].squeeze(0)
                else:
                    h_post = h_pre

                gw = gate_weights[layer_idx].to(h_post.dtype)
                gate_logits = F.linear(h_post.float(), gw.float())
                true_topk = torch.topk(gate_logits, k=min(6, gate_logits.shape[-1]), dim=-1)
                true_ids = true_topk.indices

                all_traces[layer_idx]["h_pre"].append(h_pre)
                all_traces[layer_idx]["h_post"].append(h_post)
                all_traces[layer_idx]["true_gate_ids"].append(true_ids.cpu())

        print(f"  Prompt {pi+1}/{len(prompts)}: {actual_len} tokens")

    for h in hooks:
        h.remove()

    # Save traces
    for layer_idx in anchor_layers:
        data = all_traces[layer_idx]
        if not data["h_pre"]:
            print(f"  WARNING: No data captured for layer {layer_idx}")
            continue

        h_pre_all = torch.cat(data["h_pre"], dim=0)
        h_post_all = torch.cat(data["h_post"], dim=0)
        true_ids_all = torch.cat(data["true_gate_ids"], dim=0)

        trace_path = save_dir / f"traces_layer{layer_idx}.pt"
        torch.save({
            "h_pre": h_pre_all,
            "h_post": h_post_all,
            "true_gate_ids": true_ids_all,
            "num_tokens": h_pre_all.shape[0],
            "hidden_dim": h_pre_all.shape[1],
            "num_experts": gate_weights[layer_idx].shape[0],
        }, trace_path)

        print(f"  Layer {layer_idx}: saved {h_pre_all.shape[0]} tokens, "
              f"hidden={h_pre_all.shape[1]}, experts={gate_weights[layer_idx].shape[0]}")

    # Save metadata
    meta = {
        "model": model_name,
        "anchor_layers": anchor_layers,
        "all_moe_layers": moe_layers,
        "num_prompts": args.num_prompts,
        "max_len": args.max_len,
        "hidden_dim": config.hidden_size,
        "num_experts": len(moe_layers) > 0 and gate_weights[moe_layers[0]].shape[0] or 0,
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nTraces saved to {save_dir}/")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite-Chat")
    parser.add_argument("--num-prompts", type=int, default=15)
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--save-dir", default="probes/traces")
    args = parser.parse_args()

    extract_traces(args)
