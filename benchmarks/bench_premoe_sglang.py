"""End-to-end benchmark: Pre-MoE speculative dispatch vs serial baseline.

Measures the full pipeline on 2 GPUs with real attention, gate, and FFN,
comparing serial MoE dispatch vs Pre-MoE overlapped dispatch.

Modes:
  serial:   Attn → Gate → Pack → AllToAll → FFN  (standard path)
  premoe:   Probe → [Dispatch(comm) || Attn(main)] → Gate+Verify → Fallback → FFN
  premoe_nv: Same as premoe but skip verification (upper bound)

Usage:
    torchrun --nproc_per_node=2 benchmarks/bench_premoe_sglang.py \
        --probe-dir probes --seq-lens 1024 2048 4096 8192
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent.parent))

from premoe.config import PreMoEConfig
from premoe.probe import LinearProbe, load_probes
from premoe.dispatch_planner import compute_dispatch_plan, pack_tokens, verify_dispatch
from premoe.utils import CudaTimer


# ─────────────────────────────────────────────
# Simulated operators (matching DeepSeek-V2-Lite)
# ─────────────────────────────────────────────

def run_attention(q, k, v):
    """Scaled dot product attention (simulates real attention cost)."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def run_gate(hidden, gate_w, top_k=6):
    """Gate routing: hidden @ gate_w.T -> topk indices."""
    logits = F.linear(hidden.float(), gate_w.float())
    topk = torch.topk(logits, k=top_k, dim=-1)
    return topk.indices, topk.values


def run_ffn(hidden, ffn_w):
    """Mock FFN (matmul to simulate expert compute cost)."""
    return hidden @ ffn_w


def do_send_recv(packed, send_buf, recv_buf, nccl_comm, peer, nccl_stream, comm_delay=0):
    """Pack into send_buf, do NCCL send/recv, optional delay."""
    import pre_moe_cpp
    if packed.numel() > 0:
        n = min(packed.nbytes, send_buf.nbytes)
        send_buf[:n].copy_(packed.view(-1).view(torch.uint8)[:n])
    pre_moe_cpp.async_send_recv_start(send_buf, recv_buf, nccl_comm, peer, nccl_stream)
    pre_moe_cpp.async_send_recv_wait(nccl_stream)
    if comm_delay > 0:
        torch.cuda._sleep(comm_delay)


# ─────────────────────────────────────────────
# Three benchmark modes
# ─────────────────────────────────────────────

def mode_serial(q, k, v, hidden, gate_w, ffn_w, config, rank, peer,
                send_buf, recv_buf, nccl_comm, nccl_stream, comm_delay=0):
    """Baseline: Attn → Gate → Pack → AllToAll → FFN."""
    t = CudaTimer()
    t.start()

    # Attention
    attn_out = run_attention(q, k, v)
    attn_flat = attn_out.transpose(1, 2).contiguous().view(-1, attn_out.shape[1] * attn_out.shape[-1])

    # Gate + dispatch
    true_ids, _ = run_gate(attn_flat, gate_w, config.top_k)
    peer_idx, _ = compute_dispatch_plan(true_ids, config.num_experts, config.ep_size, rank)
    packed = pack_tokens(attn_flat, peer_idx)
    do_send_recv(packed, send_buf, recv_buf, nccl_comm, peer, nccl_stream, comm_delay)

    # FFN
    _ = run_ffn(attn_flat, ffn_w)

    t.stop()
    return {"T_total": t.elapsed_ms()}


def mode_premoe(q, k, v, hidden, gate_w, probe, ffn_w, config, rank, peer,
                send_buf, recv_buf, nccl_comm, nccl_stream, comm_delay=0,
                true_ids_for_verify=None):
    """Pre-MoE: Probe → [Dispatch(comm) || Attn(main)] → Verify → Fallback → FFN."""
    t_predict = CudaTimer()
    t_overlap = CudaTimer()
    t_verify = CudaTimer()
    t_fallback = CudaTimer()
    t_total = CudaTimer()

    t_total.start()

    # Phase 1: Probe prediction
    t_predict.start()
    with torch.no_grad():
        pred_ids = torch.topk(probe(hidden), k=config.top_k, dim=-1).indices
    pred_peer_idx, _ = compute_dispatch_plan(pred_ids, config.num_experts, config.ep_size, rank)
    pred_packed = pack_tokens(hidden, pred_peer_idx)
    t_predict.stop()

    # Phase 2: Overlap — dispatch on comm stream, attention on main stream
    t_overlap.start()
    comm_stream = torch.cuda.Stream(priority=-1)
    ev = torch.cuda.Event()
    ev.record()

    with torch.cuda.stream(comm_stream):
        comm_stream.wait_event(ev)
        do_send_recv(pred_packed, send_buf, recv_buf, nccl_comm, peer, nccl_stream, comm_delay)

    # Attention on main stream (overlapping)
    attn_out = run_attention(q, k, v)
    attn_flat = attn_out.transpose(1, 2).contiguous().view(-1, attn_out.shape[1] * attn_out.shape[-1])

    torch.cuda.current_stream().wait_stream(comm_stream)
    t_overlap.stop()

    # Phase 3: Verify
    t_verify.start()
    true_ids, _ = run_gate(attn_flat, gate_w, config.top_k)
    if true_ids_for_verify is not None:
        true_ids = true_ids_for_verify
    mismatch_idx = verify_dispatch(pred_ids, true_ids, config.num_experts, config.ep_size, rank)
    n_mismatch = len(mismatch_idx)
    t_verify.stop()

    # Phase 4: Fallback
    t_fallback.start()
    if n_mismatch > 0:
        fb_packed = pack_tokens(attn_flat, mismatch_idx)
        do_send_recv(fb_packed, send_buf, recv_buf, nccl_comm, peer, nccl_stream, comm_delay)
    t_fallback.stop()

    # FFN
    _ = run_ffn(attn_flat, ffn_w)

    t_total.stop()

    return {
        "T_predict": t_predict.elapsed_ms(),
        "T_overlap": t_overlap.elapsed_ms(),
        "T_verify": t_verify.elapsed_ms(),
        "T_fallback": t_fallback.elapsed_ms(),
        "T_total": t_total.elapsed_ms(),
        "n_mismatch": n_mismatch,
        "n_total": hidden.shape[0],
        "mismatch_rate": n_mismatch / max(hidden.shape[0], 1),
    }


def mode_premoe_noverify(q, k, v, hidden, probe, ffn_w, config, rank, peer,
                         send_buf, recv_buf, nccl_comm, nccl_stream, comm_delay=0):
    """Pre-MoE without verify: Probe → [Dispatch(comm) || Attn(main)] → FFN."""
    t_predict = CudaTimer()
    t_overlap = CudaTimer()
    t_total = CudaTimer()

    t_total.start()

    t_predict.start()
    with torch.no_grad():
        pred_ids = torch.topk(probe(hidden), k=config.top_k, dim=-1).indices
    pred_peer_idx, _ = compute_dispatch_plan(pred_ids, config.num_experts, config.ep_size, rank)
    pred_packed = pack_tokens(hidden, pred_peer_idx)
    t_predict.stop()

    t_overlap.start()
    comm_stream = torch.cuda.Stream(priority=-1)
    ev = torch.cuda.Event()
    ev.record()

    with torch.cuda.stream(comm_stream):
        comm_stream.wait_event(ev)
        do_send_recv(pred_packed, send_buf, recv_buf, nccl_comm, peer, nccl_stream, comm_delay)

    attn_out = run_attention(q, k, v)
    attn_flat = attn_out.transpose(1, 2).contiguous().view(-1, attn_out.shape[1] * attn_out.shape[-1])

    torch.cuda.current_stream().wait_stream(comm_stream)
    t_overlap.stop()

    _ = run_ffn(attn_flat, ffn_w)

    t_total.stop()

    return {
        "T_predict": t_predict.elapsed_ms(),
        "T_overlap": t_overlap.elapsed_ms(),
        "T_verify": 0.0,
        "T_fallback": 0.0,
        "T_total": t_total.elapsed_ms(),
        "n_mismatch": 0,
        "n_total": hidden.shape[0],
        "mismatch_rate": 0.0,
    }


# ─────────────────────────────────────────────
# Main benchmark loop
# ─────────────────────────────────────────────

def run_benchmark(args):
    import pre_moe_cpp

    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 2))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    peer = 1 - rank

    # NCCL setup
    if rank == 0:
        nccl_id = pre_moe_cpp.get_nccl_unique_id()
    else:
        nccl_id = [0] * 128
    id_tensor = torch.tensor(nccl_id, dtype=torch.uint8).cuda()
    dist.broadcast(id_tensor, src=0)
    nccl_comm = pre_moe_cpp.create_nccl_comm(id_tensor.cpu().tolist(), rank, world_size)
    nccl_stream = pre_moe_cpp.create_cuda_stream()

    config = PreMoEConfig(probe_dir=args.probe_dir)

    # Comm delay: microseconds → GPU cycles (A100 ~1410 cycles/μs)
    comm_delay = int(args.comm_delay_us * 1410)

    # Load or create probes
    probes = load_probes(config, device)
    use_real_probes = len(probes) > 0
    if not probes:
        if rank == 0:
            print("  No trained probes found. Using random probes.")
        probes = {
            layer: LinearProbe(config.hidden_dim, config.num_experts).to(device, torch.bfloat16).eval()
            for layer in config.anchor_layers
        }
        for p in probes.values():
            for param in p.parameters():
                param.requires_grad_(False)

    # Load traces if available
    traces_dir = Path(args.traces_dir) if args.traces_dir else None
    real_traces = {}
    if traces_dir and (traces_dir / "metadata.json").exists():
        import json as _json
        with open(traces_dir / "metadata.json") as f:
            meta = _json.load(f)
        for layer_idx in config.anchor_layers:
            tp = traces_dir / f"traces_layer{layer_idx}.pt"
            if tp.exists():
                td = torch.load(tp, map_location="cpu", weights_only=True)
                real_traces[layer_idx] = {
                    "h_pre": td["h_pre"].to(torch.bfloat16),
                    "true_ids": td.get("true_gate_ids"),
                }
        if rank == 0 and real_traces:
            print(f"  Loaded real traces for layers: {list(real_traces.keys())}")

    # Shared resources
    num_heads = 16
    head_dim = config.hidden_dim // num_heads
    buf_bytes = config.comm_buffer_size_mb * 1024 * 1024
    send_buf = torch.zeros(buf_bytes, dtype=torch.uint8, device=device)
    recv_buf = torch.zeros(buf_bytes, dtype=torch.uint8, device=device)
    ffn_w = torch.randn(config.hidden_dim, config.hidden_dim, dtype=torch.bfloat16, device=device) * 0.01

    if rank == 0:
        print(f"\n  GPU: {torch.cuda.get_device_name()}")
        print(f"  Config: hidden={config.hidden_dim}, heads={num_heads}, "
              f"experts={config.num_experts}, EP={config.ep_size}, topk={config.top_k}")
        print(f"  Real probes: {use_real_probes}, Real traces: {bool(real_traces)}")
        print(f"  Comm delay: {args.comm_delay_us}μs, Warmup={args.warmup}, Iters={args.iters}")
        print()

    all_results = []

    for layer_idx, probe in probes.items():
        # Load gate weight
        gate_w = None
        if traces_dir:
            gp = traces_dir / f"gate_weight_layer{layer_idx}.pt"
            if gp.exists():
                gate_w = torch.load(gp, map_location=device, weights_only=True).to(torch.bfloat16)
        if gate_w is None:
            gate_w = torch.randn(config.num_experts, config.hidden_dim,
                                 dtype=torch.bfloat16, device=device) * 0.01

        for seq_len in args.seq_lens:
            # Prepare inputs
            if layer_idx in real_traces and real_traces[layer_idx]["h_pre"].shape[0] >= seq_len:
                hidden = real_traces[layer_idx]["h_pre"][:seq_len].to(device)
                true_ids_seq = real_traces[layer_idx]["true_ids"]
                if true_ids_seq is not None and true_ids_seq.shape[0] >= seq_len:
                    true_ids_seq = true_ids_seq[:seq_len].to(device)
                else:
                    true_ids_seq = None
            else:
                hidden = torch.randn(seq_len, config.hidden_dim, dtype=torch.bfloat16, device=device)
                true_ids_seq = None

            q = hidden.view(1, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
            k = hidden.view(1, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
            v = hidden.view(1, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()

            modes = {
                "serial": lambda: mode_serial(
                    q, k, v, hidden, gate_w, ffn_w, config, rank, peer,
                    send_buf, recv_buf, nccl_comm, nccl_stream, comm_delay),
                "premoe": lambda: mode_premoe(
                    q, k, v, hidden, gate_w, probe, ffn_w, config, rank, peer,
                    send_buf, recv_buf, nccl_comm, nccl_stream, comm_delay,
                    true_ids_for_verify=true_ids_seq),
                "premoe_nv": lambda: mode_premoe_noverify(
                    q, k, v, hidden, probe, ffn_w, config, rank, peer,
                    send_buf, recv_buf, nccl_comm, nccl_stream, comm_delay),
            }

            results_per_mode = {}

            for mode_name, mode_fn in modes.items():
                for _ in range(args.warmup):
                    mode_fn()
                torch.cuda.synchronize()

                timings = []
                detail_accum = {}
                for _ in range(args.iters):
                    r = mode_fn()
                    timings.append(r["T_total"])
                    for key, val in r.items():
                        if key != "T_total" and isinstance(val, (int, float)):
                            detail_accum.setdefault(key, []).append(val)
                torch.cuda.synchronize()

                timings.sort()
                p50 = timings[len(timings) // 2]
                p99 = timings[int(len(timings) * 0.99)]

                result = {"mode": mode_name, "layer": layer_idx, "seq_len": seq_len,
                          "T_total_p50": round(p50, 4), "T_total_p99": round(p99, 4)}
                for key, vals in detail_accum.items():
                    if isinstance(vals[0], float):
                        result[key] = round(sum(vals) / len(vals), 4)
                    else:
                        result[key] = int(sum(vals) / len(vals))
                results_per_mode[mode_name] = result

            if rank == 0:
                s = results_per_mode["serial"]
                p = results_per_mode["premoe"]
                pnv = results_per_mode["premoe_nv"]
                sp_p = s["T_total_p50"] / max(p["T_total_p50"], 0.001)
                sp_nv = s["T_total_p50"] / max(pnv["T_total_p50"], 0.001)

                print(f"  Layer {layer_idx}, SeqLen={seq_len}:")
                print(f"    Serial    : {s['T_total_p50']:.3f}ms")
                print(f"    PreMoE    : {p['T_total_p50']:.3f}ms ({sp_p:.3f}x)"
                      f"  [pred={p.get('T_predict',0):.3f} ovlp={p.get('T_overlap',0):.3f} "
                      f"verify={p.get('T_verify',0):.3f} fb={p.get('T_fallback',0):.3f} "
                      f"miss={p.get('mismatch_rate',0):.1%}]")
                print(f"    PreMoE_NV : {pnv['T_total_p50']:.3f}ms ({sp_nv:.3f}x)")

                all_results.append({
                    "layer": layer_idx, "seq_len": seq_len,
                    "comm_delay_us": args.comm_delay_us,
                    "serial": s, "premoe": p, "premoe_nv": pnv,
                    "speedup_premoe": round(sp_p, 4),
                    "speedup_premoe_nv": round(sp_nv, 4),
                })

    # Summary & save
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"  SUMMARY: Pre-MoE E2E Benchmark")
        print(f"  GPU: {torch.cuda.get_device_name()}, Delay: {args.comm_delay_us}μs")
        print(f"{'='*80}")
        print(f"  {'Layer':>5} {'SeqLen':>7} {'Serial':>8} {'PreMoE':>8} "
              f"{'NV':>8} {'Sp(PM)':>8} {'Sp(NV)':>8} {'Miss%':>6}")
        print(f"  {'-'*60}")
        for r in all_results:
            print(f"  {r['layer']:>5} {r['seq_len']:>7} "
                  f"{r['serial']['T_total_p50']:>7.3f}  "
                  f"{r['premoe']['T_total_p50']:>7.3f}  "
                  f"{r['premoe_nv']['T_total_p50']:>7.3f}  "
                  f"{r['speedup_premoe']:>7.3f}x "
                  f"{r['speedup_premoe_nv']:>7.3f}x "
                  f"{r['premoe'].get('mismatch_rate', 0):>5.1%}")
        print(f"{'='*80}")

        out = Path("results")
        out.mkdir(exist_ok=True)
        suffix = f"_delay{args.comm_delay_us}us" if args.comm_delay_us > 0 else ""
        result_file = out / f"premoe_sglang_bench{suffix}.json"
        with open(result_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Results saved to {result_file}")

    pre_moe_cpp.destroy_cuda_stream(nccl_stream)
    pre_moe_cpp.destroy_nccl_comm(nccl_comm)
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Pre-MoE E2E Benchmark")
    parser.add_argument("--probe-dir", default="probes")
    parser.add_argument("--traces-dir", default="traces",
                        help="Dir with traces + gate weights (from extract_traces.py)")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--comm-delay-us", type=int, default=0,
                        help="Simulated comm delay in μs (0=NVLink, 1000=IB, 5000=PCIe)")
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
