"""Owner's offline GPU capacity check, NOT a RAGTruth training experiment.

Uses a randomly initialized DistilBERT architecture and synthetic tensors only.
No dataset, tokenizer, downloaded checkpoint, prediction, or quality metric.
The final model/data implementations remain with their assigned teammates.
"""

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import time


ROOT = Path(__file__).resolve().parents[1]


def masked_bce(torch, logits, targets, response_mask):
    """Synthetic four-channel loss for the hardware probe only."""
    selected = response_mask.bool()
    if not selected.any():
        raise ValueError("The probe requires at least one supervised position.")
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits[selected], targets[selected]
    )


def check_mask(torch):
    # Check the loss's output positions, not encoder context gradients.
    logits = torch.zeros(1, 4, 4, requires_grad=True)
    targets = torch.zeros_like(logits)
    mask = torch.tensor([[False, True, True, False]])
    original = masked_bce(torch, logits, targets, mask)
    changed = logits.detach().clone()
    changed[~mask] = 100
    excluded_targets = targets.clone()
    excluded_targets[~mask] = 1
    assert torch.equal(original, masked_bce(torch, changed, excluded_targets, mask))
    changed[mask] = 1
    assert not torch.equal(original, masked_bce(torch, changed, targets, mask))
    original.backward()
    assert (logits.grad[~mask] == 0).all()
    assert (logits.grad[mask] != 0).all()


def run(args):
    import torch
    from transformers import DistilBertConfig, DistilBertModel

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required; no CPU fallback. Use the existing ROCm WSL environment.")
    torch.manual_seed(args.seed)
    check_mask(torch)
    device = torch.device("cuda")
    config = DistilBertConfig()
    config._attn_implementation = "eager"
    encoder = DistilBertModel(config).to(device)
    head = torch.nn.Linear(config.dim, 4).to(device)
    encoder.train()
    head.train()
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()), lr=1e-5
    )
    ids = torch.randint(999, config.vocab_size, (args.batch_size, args.sequence_length), device=device)
    attention = torch.ones_like(ids)
    attention[:, -8:] = 0
    ids[:, -8:] = config.pad_token_id
    response_mask = torch.zeros_like(ids, dtype=torch.bool)
    response_mask[:, args.sequence_length // 2:-8] = True
    targets = torch.randint(0, 2, (*ids.shape, 4), device=device).float()

    def step():
        optimizer.zero_grad(set_to_none=True)
        hidden = encoder(input_ids=ids, attention_mask=attention).last_hidden_state
        loss = masked_bce(torch, head(hidden), targets, response_mask)
        loss.backward()
        optimizer.step()
        return loss

    weight_before = head.weight.detach().clone()
    for _ in range(args.warmup_steps):
        warmup_loss = step()
    torch.cuda.synchronize()
    if not torch.isfinite(warmup_loss):
        raise RuntimeError("Non-finite warmup loss.")
    if torch.equal(weight_before, head.weight.detach()):
        raise RuntimeError("Optimizer did not change the probe head weights.")
    finite = all(
        torch.isfinite(p.grad).all().item()
        for module in (encoder, head) for p in module.parameters()
        if p.grad is not None
    )
    if not finite:
        raise RuntimeError("Non-finite warmup gradients.")

    allocated_after_warmup = torch.cuda.memory_allocated()
    reserved_after_warmup = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.steps):
        last_loss = step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if not torch.isfinite(last_loss):
        raise RuntimeError("Non-finite final loss.")
    props = torch.cuda.get_device_properties(0)
    packages = {}
    for name in ("torch", "transformers", "numpy", "tokenizers", "safetensors", "huggingface-hub"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "status": "passed",
        "kind": "synthetic_hardware_smoke_benchmark",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "Random weights and synthetic IDs/targets; no RAGTruth examples or annotations.",
            "No pretrained weights, tokenizer/alignment, real context windows, IO, evaluation, or checkpoint saving.",
            "This is not the project model, an accuracy experiment, a real-data pilot, or a full-training-time estimate.",
            "Allocator memory describes this process, not total system GPU use.",
        ],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "hip": torch.version.hip,
        "gpu": props.name,
        "gpu_total_bytes": props.total_memory,
        "architecture": {
            "name": "DistilBertModel plus disposable four-channel linear probe",
            "initialization": "random; no network or checkpoint loading",
            "layers": config.n_layers, "hidden_size": config.dim,
            "attention_heads": config.n_heads,
            "attention_implementation": encoder.config._attn_implementation,
            "parameters": sum(p.numel() for m in (encoder, head) for p in m.parameters()),
        },
        "settings": {
            "seed": args.seed, "precision": "float32", "optimizer": "AdamW",
            "learning_rate": 1e-5, "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "padding_tokens_per_sequence": 8,
            "supervised_response_tokens_per_sequence": args.sequence_length - args.sequence_length // 2 - 8,
            "warmup_optimizer_steps": args.warmup_steps,
            "measured_optimizer_steps": args.steps,
        },
        "checks": {
            "response_loss_mask": "passed", "forward_backward_optimizer": "passed",
            "finite_warmup_gradients": True, "head_weights_changed": True,
            "final_loss_finite": True,
        },
        "measurement": {
            "elapsed_seconds": elapsed,
            "seconds_per_step": elapsed / args.steps,
            "synthetic_sequences_per_second": args.batch_size * args.steps / elapsed,
            "synthetic_input_positions_per_second": args.batch_size * args.sequence_length * args.steps / elapsed,
            "allocated_bytes_after_warmup": allocated_after_warmup,
            "reserved_bytes_after_warmup": reserved_after_warmup,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 32 <= args.sequence_length <= 512:
        parser.error("sequence-length must be 32..512 for this bounded hardware probe")
    if not 1 <= args.batch_size <= 4 or not 1 <= args.steps <= 20 or not 1 <= args.warmup_steps <= 5:
        parser.error("bounded probe: batch-size 1..4, steps 1..20, warmup-steps 1..5")
    output = args.output or ROOT / "outputs" / (
        "gpu_smoke_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"
    )
    if output.exists():
        parser.error(f"Refusing to overwrite an existing report: {output}")
    report = run(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "kind": report["kind"], "gpu": report["gpu"],
        "settings": report["settings"], "measurement": report["measurement"],
        "report": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
