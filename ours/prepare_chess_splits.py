"""Prepare disjoint Chess pilot splits through the unchanged WHALE converter.

Pins the public dataset and excludes every previously inspected engineering ID
and corresponding board-position group. Selection never depends on model scores.
This is shared experimental data preparation, not an additional Method component.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

from .preflight import PIN
from .prepare_chess_runtime import DATASET, REVISION
from .visual_task import file_sha256


def main(args):
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq
    from autoharness_chess_puzzle import grpo_dataset

    upstream = Path("upstream/WHALE")
    if subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip() != PIN:
        raise ValueError("Wrong upstream revision")
    if subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip():
        raise ValueError("Modified upstream source")
    if not all(1 <= n <= 4096 for n in (args.train, args.mh_val, args.test)):
        raise ValueError("Pilot split sizes must be within1..4096")
    engineering = Path("data/chess-engineering-v3")
    manifest = json.loads((engineering / "manifest.json").read_text())
    source = engineering / "source_rows.jsonl"
    if file_sha256(source) != manifest["source_rows_sha256"]:
        raise ValueError("Changed engineering exclusion evidence")
    excluded_ids = set(manifest["future_split_exclusions"])
    raw_engineering = [json.loads(line) for line in source.read_text().splitlines()]
    if excluded_ids != {str(row["PuzzleId"]) for row in raw_engineering} or len(excluded_ids) != 64:
        raise ValueError("All64 inspected source IDs must be excluded")
    excluded_positions = {record["extra_info"]["position_key"] for row in raw_engineering
                          if (record := grpo_dataset.convert_lichess_row(row)) is not None}
    stats = {"source_rows_seen": 0, "excluded_id": 0, "excluded_position": 0}
    source_file = f"datasets/{DATASET}@{REVISION}/data/train-00000-of-00003.parquet"

    def pinned_stream():
        fs = HfFileSystem(token=False)
        with fs.open(source_file, "rb", block_size=1024 * 1024) as stream:
            with pq.ParquetFile(stream, pre_buffer=False) as parquet:
                for group in range(parquet.num_row_groups):
                    rows = parquet.read_row_group(group, use_threads=False).to_pylist()
                    for row in rows:
                        if stats["source_rows_seen"] >= 16384:
                            return
                        stats["source_rows_seen"] += 1
                        if str(row["PuzzleId"]) in excluded_ids:
                            stats["excluded_id"] += 1
                            continue
                        record = grpo_dataset.convert_lichess_row(row)
                        if record is not None and record["extra_info"]["position_key"] in excluded_positions:
                            stats["excluded_position"] += 1
                            continue
                        yield row

    def bound_loader(name, *, split, streaming):
        if name != DATASET or split != "train" or streaming is not True:
            raise ValueError("Unexpected native dataset request")
        return pinned_stream()

    args.output.mkdir(parents=True, exist_ok=False)
    with patch("datasets.load_dataset", bound_loader):
        paths = grpo_dataset.build_splits(output_dir=args.output, dataset_name=DATASET,
                    train_size=args.train, test_size=args.test, mh_val_size=args.mh_val,
                    seed=42, max_scan=16384, min_solution_plies=1, max_solution_plies=9,
                    min_rating=None, max_rating=None, preserve_existing_eval_splits=False)
    seen_positions, seen_ids, splits = set(), set(), {}
    for name, path in paths.items():
        parquet = pq.ParquetFile(path, pre_buffer=False)
        records = [row for i in range(parquet.num_row_groups)
                   for row in parquet.read_row_group(i, use_threads=False).to_pylist()]
        ids = [row["extra_info"]["puzzle_id"] for row in records]
        positions = [row["extra_info"]["position_key"] for row in records]
        assert len(ids) == len(set(ids)) == len(set(positions))
        assert not set(ids) & (seen_ids | excluded_ids)
        assert not set(positions) & (seen_positions | excluded_positions)
        seen_ids.update(ids)
        seen_positions.update(positions)
        splits[name] = {"path": str(path), "examples": len(records), "sha256": file_sha256(path),
                        "puzzle_ids": ids, "position_keys": positions}
    report = {"kind": "native_chess_pilot_splits", "status": "PREPARED_AND_DISJOINT", "role": "pilot",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "upstream_revision": PIN,
              "dataset": DATASET, "revision": REVISION, "source_file": source_file, "split_seed": 42,
              "source_scan": stats, "excluded_ids": sorted(excluded_ids),
              "excluded_positions": sorted(excluded_positions), "splits": splits,
              "source_sha256": {str(p): file_sha256(p) for p in
                  [Path(__file__), Path(grpo_dataset.__file__), source, engineering / "manifest.json"]},
              "model_calls": 0, "limitations": ["Small pilot sizes, not the original paper splits or paper accuracy",
                  "Same upstream converter, split hash and fallback filling rule; pinned source and engineering exclusions are declared differences",
                  "No model-based filtering, evaluation, training or F0 scientific pass; test content remains outside proposer access"]}
    (args.output / "ours-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "source_scan": stats,
                      "sizes": {name: value["examples"] for name, value in splits.items()}, "model_calls": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=int, default=128)
    parser.add_argument("--mh-val", type=int, default=32)
    parser.add_argument("--test", type=int, default=64)
    main(parser.parse_args())
