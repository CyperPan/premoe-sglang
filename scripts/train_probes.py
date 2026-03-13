"""Step 2: Train per-layer Linear Probes on extracted traces.

For each anchor layer, trains a simple Linear(hidden_dim, num_experts)
probe that predicts expert routing from pre-attention hidden states.

Usage:
    python scripts/train_probes.py [--traces-dir probes/traces] [--topk 6]
"""
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class LinearProbe(nn.Module):
    """Minimal linear probe: hidden_dim -> num_experts."""
    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, x):
        return self.linear(x)


def train_probe_for_layer(traces_path, gate_weight_path, topk, ep_size, save_path):
    """Train and evaluate a probe for one layer."""
    print(f"\n{'='*60}")
    print(f"  Training probe: {Path(traces_path).stem}")
    print(f"{'='*60}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = torch.load(traces_path, map_location="cpu", weights_only=True)
    h_pre = data["h_pre"].float()
    true_ids = data["true_gate_ids"]
    num_tokens = data["num_tokens"]
    hidden_dim = data["hidden_dim"]
    num_experts = data["num_experts"]

    if num_experts > 512:
        print(f"  SKIPPING: {num_experts} experts too many")
        return {"skipped": True}

    gate_w = torch.load(gate_weight_path, map_location="cpu", weights_only=True)

    split = int(num_tokens * 0.8)
    train_h, test_h = h_pre[:split].to(device), h_pre[split:].to(device)
    train_ids, test_ids = true_ids[:split], true_ids[split:]

    def make_targets(ids, n_exp):
        N = ids.shape[0]
        targets = torch.zeros(N, n_exp)
        for k in range(ids.shape[1]):
            targets.scatter_(1, ids[:, k:k+1].long(), 1.0)
        return targets

    train_targets = make_targets(train_ids, num_experts).to(device)
    test_targets = make_targets(test_ids, num_experts).to(device)

    probe = LinearProbe(hidden_dim, num_experts).to(device)
    with torch.no_grad():
        probe.linear.weight.copy_(gate_w.float())

    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-5)
    batch_size = 4096
    best_acc = 0.0

    for epoch in range(30):
        probe.train()
        perm = torch.randperm(train_h.shape[0], device=device)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, train_h.shape[0], batch_size):
            idx = perm[i:i+batch_size]
            logits = probe(train_h[idx])
            loss = F.binary_cross_entropy_with_logits(logits, train_targets[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0:
            probe.eval()
            with torch.no_grad():
                test_logits = probe(test_h)
                pred_topk = torch.topk(test_logits, k=topk, dim=-1).indices
                test_ids_dev = test_ids.to(device)

                total_recall = 0.0
                for t in range(test_h.shape[0]):
                    ps = set(pred_topk[t].cpu().tolist())
                    ts = set(test_ids[t].tolist())
                    total_recall += len(ps & ts) / max(len(ts), 1)
                avg_recall = total_recall / test_h.shape[0]

                experts_per_gpu = num_experts // ep_size
                pred_gpus = pred_topk // experts_per_gpu
                true_gpus = test_ids_dev.long() // experts_per_gpu

                gpu_match = 0
                for t in range(test_h.shape[0]):
                    pg = set(pred_gpus[t].cpu().tolist())
                    tg = set(true_gpus[t].cpu().tolist())
                    if pg == tg:
                        gpu_match += 1
                gpu_acc = gpu_match / test_h.shape[0]

            print(f"  Epoch {epoch+1:2d}: loss={total_loss/n_batches:.4f}, "
                  f"top-{topk} recall={avg_recall:.4f}, GPU-acc={gpu_acc:.4f}")

            if avg_recall > best_acc:
                best_acc = avg_recall
                torch.save(probe.state_dict(), save_path)

    # Final evaluation
    probe.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    probe.eval()

    with torch.no_grad():
        test_logits = probe(test_h)
        pred_topk = torch.topk(test_logits, k=topk, dim=-1).indices
        pred_topk_cpu = pred_topk.cpu()

        recalls = []
        for t in range(test_h.shape[0]):
            ps = set(pred_topk_cpu[t].tolist())
            ts = set(test_ids[t].tolist())
            recalls.append(len(ps & ts) / max(len(ts), 1))
        recalls = torch.tensor(recalls)

        experts_per_gpu = num_experts // ep_size
        pred_gpus = pred_topk_cpu // experts_per_gpu
        true_gpus = test_ids.long() // experts_per_gpu

        gpu_matches = []
        for t in range(test_h.shape[0]):
            pg = set(pred_gpus[t].tolist())
            tg = set(true_gpus[t].tolist())
            gpu_matches.append(1.0 if pg == tg else 0.0)
        gpu_matches = torch.tensor(gpu_matches)

    results = {
        "expert_topk_recall": recalls.mean().item(),
        "gpu_dispatch_accuracy": gpu_matches.mean().item(),
        "gpu_miss_rate": 1.0 - gpu_matches.mean().item(),
        "total_test_tokens": test_h.shape[0],
    }

    print(f"\n  Final: recall={results['expert_topk_recall']:.4f}, "
          f"GPU-acc={results['gpu_dispatch_accuracy']:.4f}")

    del train_h, test_h, train_targets, test_targets
    probe.cpu()
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-dir", default="probes/traces")
    parser.add_argument("--probes-dir", default="probes")
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--ep-size", type=int, default=2)
    args = parser.parse_args()

    traces_dir = Path(args.traces_dir)
    probes_dir = Path(args.probes_dir)
    probes_dir.mkdir(parents=True, exist_ok=True)

    with open(traces_dir / "metadata.json") as f:
        meta = json.load(f)

    anchor_layers = meta["anchor_layers"]
    print(f"Model: {meta['model']}")
    print(f"Anchor layers: {anchor_layers}")

    all_results = {}
    for layer_idx in anchor_layers:
        traces_path = traces_dir / f"traces_layer{layer_idx}.pt"
        gate_path = traces_dir / f"gate_weight_layer{layer_idx}.pt"
        probe_path = probes_dir / f"probe_layer{layer_idx}.pt"

        if not traces_path.exists():
            continue

        results = train_probe_for_layer(
            str(traces_path), str(gate_path),
            args.topk, args.ep_size, str(probe_path)
        )
        all_results[f"layer_{layer_idx}"] = results

    summary_path = probes_dir / "probe_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
