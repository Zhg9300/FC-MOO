#!/usr/bin/env python3
"""Download Banking77, generate client partitions, and verify the prepared data."""

import argparse
import json
from pathlib import Path

import numpy as np

from fl_exp.data import (
    BASE_PARTITIONS, EXTRA_PARTITIONS, RAW_FILES, REVISION, SUBSET_PARTITIONS,
    ensure_raw, expand_pools, load_raw, make_full_partition, make_partition, sample_pools,
    save_partition, sha256, verify_prepared, write_csv, write_json, write_jsonl,
)


def prepare_extra_partition(output_dir, raw_dir, seed, name):
    """Add a partition without changing the manifest of the original data."""
    verify_prepared(output_dir, raw_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["seed"] != seed or manifest["sampling"].get("full_training"):
        raise ValueError("Expected subset data with the same seed")
    destination = output_dir / name
    if destination.exists():
        raise ValueError("Partition already exists; use --verify-only: " + str(destination))
    pools = {split: [json.loads(line) for line in (
        output_dir / "pools" / (split + ".jsonl")
    ).read_text(encoding="utf-8").splitlines()] for split in ("train", "validation", "test")}
    labels = json.loads((output_dir / "labels.json").read_text(encoding="utf-8"))
    clients, metadata = make_partition(pools, seed, name)
    save_partition(destination, clients, labels, metadata)
    shared = ["manifest.json", "labels.json", "labels.csv", "sampling_audit.json"]
    shared += ["pools/" + split + ".jsonl" for split in pools]
    write_json(destination / "manifest.json", {
        "name": name, "seed": seed,
        "reference_files": {path: sha256(output_dir / path) for path in shared},
        "files": [{"path": str(path.relative_to(destination)), "sha256": sha256(path)}
                  for path in sorted(destination.rglob("*")) if path.is_file()],
    })
    validation = verify_prepared(output_dir, raw_dir)
    write_json(destination / "validation.json", validation)
    print(json.dumps({"output_dir": str(destination), **validation}, ensure_ascii=False, indent=2))


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--raw-dir", type=Path, default=root / "data/raw/banking77")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--partition", choices=SUBSET_PARTITIONS,
                        help="Prepare one subset partition; default: Dirichlet alpha=0.1 and alpha=0.2")
    parser.add_argument("--full", action="store_true", help="Append all eligible training rows; keep the subset held-out assignments")
    parser.add_argument("--reference-dir", type=Path, help="Existing subset seed directory used by --full")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if args.reference_dir and not args.full:
        parser.error("--reference-dir requires --full")
    if args.partition and args.full:
        parser.error("--partition is only supported for subset preparation")
    dataset_root = "banking77_full" if args.full else "banking77"
    output_dir = args.output_dir or root / "data/prepared" / dataset_root / ("seed_" + str(args.seed))
    if args.verify_only:
        print(json.dumps(verify_prepared(output_dir, args.raw_dir), ensure_ascii=False, indent=2))
        return
    if args.partition in EXTRA_PARTITIONS:
        if not (output_dir / "manifest.json").is_file():
            parser.error("Prepare the default subset first, then add --partition " + args.partition)
        prepare_extra_partition(output_dir, args.raw_dir, args.seed, args.partition)
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error("output directory is not empty; use --verify-only or a new --output-dir")
    ensure_raw(args.raw_dir)
    labels, raw = load_raw(args.raw_dir)
    reference_dir = args.reference_dir or root / "data/prepared/banking77" / ("seed_" + str(args.seed))
    if args.full:
        verify_prepared(reference_dir, args.raw_dir)
        reference = json.loads((reference_dir / "manifest.json").read_text(encoding="utf-8"))
        if reference["seed"] != args.seed or reference["partitions"] != list(BASE_PARTITIONS):
            parser.error("--full requires the default subset partitions with the same seed")
        reference_pools = {split: [json.loads(line) for line in (
            reference_dir / "pools" / (split + ".jsonl")
        ).read_text(encoding="utf-8").splitlines()] for split in ("train", "validation", "test")}
        pools, audit = expand_pools(raw, reference_pools)
        audit["reference_manifest_sha256"] = sha256(reference_dir / "manifest.json")
    else:
        pools, audit = sample_pools(raw, args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "labels.json", labels)
    write_csv(output_dir / "labels.csv", ["label", "label_name"],
              [{"label": index, "label_name": label} for index, label in enumerate(labels)])
    write_json(output_dir / "sampling_audit.json", audit)
    for split, records in pools.items():
        write_jsonl(output_dir / "pools" / (split + ".jsonl"), records)
    partitions = []
    reference_evaluation, reference_training = {}, {}
    for name in ((args.partition,) if args.partition else BASE_PARTITIONS):
        if args.full:
            source = reference_dir / name
            previous_clients = [{split: [json.loads(line) for line in (
                source / "clients" / ("client_{:02d}".format(client)) / (split + ".jsonl")
            ).read_text(encoding="utf-8").splitlines()] for split in pools} for client in range(10)]
            previous_metadata = json.loads((source / "partition.json").read_text(encoding="utf-8"))
            clients, metadata = make_full_partition(pools, args.seed, previous_clients, previous_metadata)
            metadata["reference_partition_sha256"] = sha256(source / "partition.json")
            for client in range(10):
                relative = Path(name + "_full") / "clients" / ("client_{:02d}".format(client))
                original = source / "clients" / ("client_{:02d}".format(client))
                reference_training[str(relative / "train.jsonl")] = {
                    "rows": len(previous_clients[client]["train"]), "sha256": sha256(original / "train.jsonl")}
                for split in ("validation", "test"):
                    reference_evaluation[str(relative / (split + ".jsonl"))] = sha256(original / (split + ".jsonl"))
        else:
            clients, metadata = make_partition(pools, args.seed, name)
        actual_name = metadata["name"]
        save_partition(output_dir / actual_name, clients, labels, metadata)
        partitions.append(actual_name)
        print(actual_name, "train counts:", [len(client["train"]) for client in clients], flush=True)
    files = [{"path": str(path.relative_to(output_dir)), "sha256": sha256(path)}
             for path in sorted(output_dir.rglob("*")) if path.is_file()]
    manifest = {
        "dataset": "Banking77", "upstream_revision": REVISION, "seed": args.seed,
        "numpy_version": np.__version__, "rng": "numpy PCG64 / SeedSequence independent streams",
        "n_clients": 10, "n_labels": 77, "partitions": partitions,
        "sample_counts": {split: len(records) for split, records in pools.items()},
        "sampling": {"max_train_per_label": 50, "validation_per_label": 10,
                     "official_test_preserved": True,
                     "underfilled_labels": [item for item in audit["per_label_allocation"] if item["train"] < 50]},
        "raw_sha256": {name: sha256(args.raw_dir / name) for name in RAW_FILES},
        "files": files,
    }
    if args.full:
        manifest["sampling"] = dict(full_training=True, max_train_per_label=None,
                                    validation_per_label=10, official_test_preserved=True,
                                    reference_assignments_preserved=True)
        manifest["reference"] = dict(directory=str(reference_dir.resolve()),
                                     manifest_sha256=sha256(reference_dir / "manifest.json"))
        for split in ("validation", "test"):
            reference_evaluation[str(Path("pools") / (split + ".jsonl"))] = sha256(reference_dir / "pools" / (split + ".jsonl"))
        manifest["reference_evaluation_sha256"] = reference_evaluation
        manifest["reference_training_prefixes"] = reference_training
    write_json(output_dir / "manifest.json", manifest)
    validation = verify_prepared(output_dir, args.raw_dir)
    write_json(output_dir / "validation.json", validation)
    print(json.dumps({"output_dir": str(output_dir), **validation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
