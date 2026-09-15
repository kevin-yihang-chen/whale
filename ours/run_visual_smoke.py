"""Real cached-Qwen engineering check of the image -> target -> E1 connection.

This is not WHALE reproduction, an optimized harness, training, or evidence of
VETO gains. Thirty-two greedy generations are a fixed input-contract experiment.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import time

from PIL import Image

from .evidence import EvaluationIdentity, PairPrediction, VisualPair, VisualPairEvaluator, fingerprint
from .visual_task import SYSTEM_PROMPT, VisualTaskAdapter, binary_answer_verifier, file_sha256


def load_inputs(manifest_path: Path):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("role") != "engineering" or manifest.get("kind") != "rendered_input_fixture":
        raise ValueError("This smoke command accepts engineering fixtures only")
    pairs = tuple(VisualPair(**p) for p in manifest["pairs"])
    if len(pairs) != 8 or fingerprint([asdict(p) for p in pairs]) != manifest["audit_data_sha256"]:
        raise ValueError("Expected the fixed eight-pair engineering manifest")
    root = manifest_path.parent.resolve()
    paths = {}
    for sha, relative in manifest["image_files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != sha:
            raise ValueError("Image path or content does not match the manifest")
        paths[sha] = path
    if set(paths) != {s for p in pairs for s in p.images_sha256}:
        raise ValueError("Image coverage differs from the pair manifest")
    return manifest, pairs, paths


def run(args) -> None:
    # Set offline behavior before importing the HF stack. Credentials are never logged.
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        os.environ.pop(key, None)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    torch.set_num_threads(4)
    start = time.perf_counter()
    manifest, pairs, paths = load_inputs(args.manifest)
    model_path = args.model.resolve()
    if not model_path.is_dir():
        raise ValueError("An existing local model snapshot is required")
    config = json.loads((model_path / "config.json").read_text())
    if config.get("model_type") != "qwen2_5_vl":
        raise ValueError("This engineering driver requires the cached Qwen2.5-VL model")
    weights = sorted(model_path.glob("*.safetensors"))
    if not weights:
        raise ValueError("No model weight shards found")
    args.output.mkdir(parents=True, exist_ok=False)
    source_files = [Path(__file__), Path(__file__).with_name("visual_task.py"), Path(__file__).with_name("evidence.py")]
    source_hashes = {p.name: file_sha256(p) for p in source_files}
    adapter = VisualTaskAdapter()
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
                                             min_pixels=adapter.min_pixels, max_pixels=adapter.max_pixels)
    input_records = []
    requests = []
    image_token_id = config["image_token_id"]
    for pair in pairs:
        for mode in ("visual", "no_image"):
            for side, sha in enumerate(pair.images_sha256):
                with Image.open(paths[sha]) as image:
                    tensors, prompt = adapter.prepare(processor, question=pair.question,
                                                       image=image if mode == "visual" else None)
                tokens = int((tensors["input_ids"] == image_token_id).sum())
                if (mode == "visual") != (tokens > 0):
                    raise ValueError("Processor image-token presence differs from the requested mode")
                if mode == "visual":
                    grid = tensors["image_grid_thw"]
                    expected = int(grid.prod()) // processor.image_processor.merge_size ** 2
                    if tokens != expected or "pixel_values" not in tensors:
                        raise ValueError("Visual token/grid/pixel inputs disagree")
                record = {"pair_id": pair.pair_id, "side": side, "mode": mode,
                          "image_sha256": sha if mode == "visual" else None,
                          "prompt_sha256": fingerprint(prompt), "input_ids_sha256": fingerprint(tensors["input_ids"].tolist()),
                          "input_tokens": int(tensors["input_ids"].numel()), "visual_tokens": tokens}
                requests.append((record, tensors))
                input_records.append(record)
    setup = {"kind": "real_processor_preflight", "role": "engineering", "model_path": str(model_path),
             "manifest_file_sha256": file_sha256(args.manifest), "manifest_content_sha256": manifest["audit_data_sha256"],
             "source_sha256": source_hashes, "input_records": input_records,
             "versions": {name: version(name) for name in ("torch", "transformers", "Pillow")}}
    (args.output / "processor.json").write_text(json.dumps(setup, indent=2) + "\n")
    if args.preflight_only:
        print(json.dumps({"status": "PROCESSOR_PASS", "requests": len(requests), "model_generations": 0}))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Model generation requires an allocated CUDA GPU")
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    weight_hashes = {p.name: file_sha256(p) for p in weights}
    asset_hashes = {p.name: file_sha256(p) for p in model_path.iterdir()
                    if p.is_file() and p.suffix in {".json", ".jinja", ".txt"}}
    decode = {"do_sample": False, "max_new_tokens": 8, "dtype": "bfloat16", "attention": "sdpa", "seed": 17,
              "min_pixels": adapter.min_pixels, "max_pixels": adapter.max_pixels}
    identity = EvaluationIdentity(fingerprint({"weights": weight_hashes, "assets": asset_hashes}), fingerprint([]),
                                  manifest["audit_data_sha256"], fingerprint(decode), source_hashes["visual_task.py"],
                                  "engineering-frozen-pretrained")
    load_start = time.perf_counter()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path), dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True).to("cuda").eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    torch.cuda.reset_peak_memory_stats()
    completed = []
    with (args.output / "generations.jsonl").open("x") as ledger:
        for record, cpu_tensors in requests:
            tensors = cpu_tensors.to("cuda")
            torch.cuda.synchronize()
            call_start = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**tensors, do_sample=False, max_new_tokens=8,
                                           use_cache=True, pad_token_id=processor.tokenizer.pad_token_id)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - call_start
            suffix = generated[0, tensors["input_ids"].shape[1]:]
            raw = processor.tokenizer.decode(suffix, skip_special_tokens=True)
            eos_ids = model.generation_config.eos_token_id
            eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
            ended_on_eos = bool(suffix.numel()) and int(suffix[-1]) in eos_ids
            row = {**record, "raw_answer": raw, "output_tokens": int(suffix.numel()),
                   "generation_seconds": elapsed, "finish": "eos" if ended_on_eos else "length"}
            ledger.write(json.dumps(row) + "\n")
            ledger.flush()
            completed.append(row)
            del tensors, generated, suffix
            print(f"completed {len(completed)}/{len(requests)}", flush=True)
    results = {}
    for mode in ("visual", "no_image"):
        answers = {(r["pair_id"], r["side"]): r["raw_answer"] for r in completed if r["mode"] == mode}
        predictions = [PairPrediction(p.pair_id, (answers[(p.pair_id, 0)], answers[(p.pair_id, 1)])) for p in pairs]
        receipt = VisualPairEvaluator(binary_answer_verifier).evaluate(
            pairs, predictions, identity=replace(identity, decode_sha256=fingerprint({**decode, "mode": mode})),
            harness_sha256=fingerprint({"system": SYSTEM_PROMPT, "adapter": source_hashes["visual_task.py"]}), role="engineering")
        results[mode] = {"receipt": asdict(receipt), "paired_accuracy": receipt.paired_accuracy,
                         "marginal_accuracy": receipt.marginal_accuracy}
    result = {**setup, "kind": "real_vlm_engineering_smoke", "status": "COMPLETED", "identity": asdict(identity),
              "completed_at_utc": datetime.now(timezone.utc).isoformat(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "gpu": torch.cuda.get_device_name(), "weight_files_sha256": weight_hashes,
              "model_assets_sha256": asset_hashes, "decode": decode, "generations": len(completed),
              "generations_sha256": file_sha256(args.output / "generations.jsonl"),
              "model_load_seconds": load_seconds, "total_seconds": time.perf_counter() - start,
              "generation_seconds": sum(r["generation_seconds"] for r in completed),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "results": results, "research_claim": "INPUT_CONNECTION_ONLY_NO_WHALE_OR_VETO_PERFORMANCE_CLAIM"}
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "generations": len(completed),
                      "result": str(args.output / "result.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    run(parser.parse_args())
