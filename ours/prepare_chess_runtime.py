"""Prepare a pinned, small Lichess input sample using the released converter.

Shared baseline infrastructure, outside Method E1-E4. Engineering examples are
explicitly excluded from future scientific splits; no model-based filtering.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .visual_task import file_sha256

DATASET = "Lichess/chess-puzzles"
REVISION = "479ea9bc9f681385f5adb23fa27a96c2dc8ae599"


def main(args):
    from huggingface_hub import HfFileSystem
    import pandas as pd
    import pyarrow.parquet as pq
    from autoharness_chess_puzzle import grpo_dataset, runner

    if args.count < 1 or args.count > 64:
        raise ValueError("This engineering command accepts at most 64 examples")
    args.output.mkdir(parents=True, exist_ok=False)
    source_file = f"datasets/{DATASET}@{REVISION}/data/train-00000-of-00003.parquet"
    raw = []
    # Synchronous row-group reads avoid the Arrow Dataset scanner's teardown
    # failure observed in this environment. Source rows and ordering stay fixed.
    fs = HfFileSystem(token=False)
    with fs.open(source_file, "rb", block_size=1024 * 1024) as stream:
        with pq.ParquetFile(stream, pre_buffer=False) as parquet:
            for group in range(parquet.num_row_groups):
                raw.extend(parquet.read_row_group(group, use_threads=False).slice(0, 64 - len(raw)).to_pylist())
                if len(raw) == 64:
                    break
    if len(raw) != 64:
        raise ValueError("Expected 64 source rows from the pinned first parquet shard")
    with (args.output / "source_rows.jsonl").open("x") as stream:
        for row in raw:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    converted = []
    positions = set()
    for row in raw:
        record = grpo_dataset.convert_lichess_row(dict(row))
        if record is None:
            continue
        info = record["extra_info"]
        if not 1 <= len(info["solution_moves"]) <= 9 or info["position_key"] in positions:
            continue
        if info["start_fen"] not in record["prompt"][-1]["content"]:
            raise ValueError("Upstream converter fell back to a prompt without board evidence")
        info["split"] = "engineering"
        positions.add(info["position_key"])
        converted.append(record)
        if len(converted) == args.count:
            break
    if len(converted) != args.count:
        raise ValueError("The fixed first 64 source rows did not fill the requested sample")
    output = args.output / "engineering.parquet"
    pd.DataFrame(converted).to_parquet(output, index=False)
    examples = runner._read_examples(output, limit=args.count, seed=42)
    if len(examples) != args.count:
        raise ValueError("Released runner could not read the prepared sample")
    report = {"kind": "pinned_lichess_engineering_inputs", "role": "engineering",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "dataset": DATASET, "revision": REVISION,
              "source_file": source_file,
              "selection": "First unique valid <=9-ply examples among first64 source rows, before inference",
              "source_rows": len(raw), "examples": len(examples),
              "puzzle_ids": [record["extra_info"]["puzzle_id"] for record in converted],
              "future_split_exclusions": [str(row["PuzzleId"]) for row in raw],
              "source_rows_sha256": file_sha256(args.output / "source_rows.jsonl"),
              "parquet_sha256": file_sha256(output),
              "converter_sha256": file_sha256(Path(grpo_dataset.__file__)),
              "runner_sha256": file_sha256(Path(runner.__file__)),
              "model_calls": 0, "limitation": "Not the paper split or a model performance measurement"}
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "INPUTS_READY", "examples": len(examples), "path": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    main(parser.parse_args())
