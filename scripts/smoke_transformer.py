"""One CPU optimizer step on two synthetic answers with LOCAL pretrained ModernBERT.

No dataset is read, no network requests are made, no epoch training is started.
Download the pinned weights separately; pass their directory with --model-dir.
All outputs must be placed under the ignored artifacts/ directory.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/pretrained-cpu-smoke")
    args = parser.parse_args(argv)
    model_dir, output = args.model_dir.resolve(), args.output.resolve()
    if not (model_dir / "model.safetensors").is_file():
        parser.error("--model-dir must contain the previously downloaded model.safetensors")
    if not output.is_relative_to(ROOT / "artifacts") or output == ROOT / "artifacts":
        parser.error("--output must be a new directory underneath artifacts/")
    if output.exists():
        parser.error("refusing to overwrite an existing smoke run")

    import torch
    from safetensors import safe_open
    import transformer as tr
    from ragtruth_data.data import sha256_file

    torch.set_num_threads(4)
    tr.seed_everything(42)
    config = json.loads((ROOT / "configs/transformer.json").read_text(encoding="utf-8"))
    config.update(model_name=str(model_dir), max_length=64, min_context_tokens=16,
                  response_overlap=8, batch_size=2, gradient_accumulation_steps=1,
                  epochs=1, precision="fp32", device="cpu")
    tr.validate_config(config)
    tokenizer = tr.AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    started = time.monotonic()
    model = tr.HallucinationTransformer.from_pretrained(config)
    tr.validate_tokenizer(tokenizer, config, model.encoder.config)
    if model.encoder.config.model_type != "modernbert":
        raise ValueError("this smoke check requires ModernBERT")
    # Check actual loaded pretrained tensors, rather than just trusting the model name.
    with safe_open(model_dir / "model.safetensors", framework="pt") as weights:
        torch.testing.assert_close(model.encoder.embeddings.tok_embeddings.weight,
                                   weights.get_tensor("model.embeddings.tok_embeddings.weight"), rtol=0, atol=0)
        torch.testing.assert_close(model.encoder.layers[-1].attn.Wqkv.weight,
                                   weights.get_tensor(f"model.layers.{len(model.encoder.layers)-1}.attn.Wqkv.weight"),
                                   rtol=0, atol=0)
    answer = "The shop opened in 1999 and serves Indian food."
    labels = []
    for text, kind in zip(("opened in 1999", "opened", "1999", "Indian food"), tr.LABEL_TYPES):
        start = answer.index(text)
        labels.append({"start": start, "end": start + len(text), "label_type": kind})
    records = [{"id": "synthetic-overlap", "source_text": "The shop opened in 2004.",
                "response": answer, "labels": labels},
               {"id": "synthetic-negative", "source_text": "A shop.", "response": "A shop.", "labels": []}]
    features = [f for row in records for f in tr.record_features(row, tokenizer, config)]
    batch = tr.collate_features(features, tokenizer.pad_token_id)
    active = batch["response_mask"].bool() & batch["attention_mask"].bool()
    assert any(target.tolist() == [1, 0, 1, 0] for target in batch["labels"][active])
    model.train()
    before = model.classifier.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    result = model(**batch)
    result["logits"].retain_grad()
    loss = result["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert (result["logits"].grad[~active] == 0).all()
    gradients = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
    assert any(torch.count_nonzero(grad) for grad in gradients)
    optimizer.step()
    assert not torch.equal(before, model.classifier.weight)
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.inference_mode():
        expected = model(**batch)["logits"].clone()
    predictions = list(tr.predict_records(model, records, tokenizer, config, torch.device("cpu")))
    for record, prediction in zip(records, predictions):
        assert set(prediction) == {"id", "spans"}
        assert prediction["id"] == record["id"]
        assert all(0 <= s["start"] < s["end"] <= len(record["response"]) and s["label_type"] in tr.LABEL_TYPES
                   for s in prediction["spans"])
    output.mkdir(parents=True)
    tr.save_checkpoint(output / "checkpoint", model, tokenizer, config, {"synthetic_smoke": True})
    encoder_layers = model.encoder.config.num_hidden_layers
    encoder_parameters = sum(p.numel() for p in model.encoder.parameters())
    loss_value = loss.item()
    del optimizer, model, result, loss, gradients
    gc.collect()
    loaded, _, _ = tr.load_checkpoint(output / "checkpoint")
    with torch.inference_mode():
        torch.testing.assert_close(expected, loaded(**batch)["logits"], rtol=0, atol=0)
    report = {"status": "passed", "pretrained_model": "answerdotai/ModernBERT-base",
              "revision": config["revision"], "weights_sha256": sha256_file(model_dir / "model.safetensors"),
              "torch": torch.__version__, "device": "cpu", "precision": "fp32",
              "encoder_layers": encoder_layers, "encoder_parameters": encoder_parameters,
              "batch_shape": list(batch["input_ids"].shape), "logits_shape": list(expected.shape),
              "answer_tokens": int(active.sum()), "loss": loss_value, "optimizer_steps": 1,
              "encoder_gradients_finite_and_nonzero": True, "masked_logit_gradients_zero": True,
              "checkpoint_logits_exact_match": True, "seconds": time.monotonic() - started,
              "scope": "two synthetic responses; no full training or quality evaluation"}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "predictions.jsonl").write_text("".join(json.dumps(p) + "\n" for p in predictions), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
