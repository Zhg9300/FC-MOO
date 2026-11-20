"""Reproducible Banking77 sampling, client partitions, and distribution audits."""

import csv
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen

import numpy as np


REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
SOURCE_BASE = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    + REVISION + "/"
)
RAW_FILES = {
    "train.csv": "banking_data/train.csv",
    "test.csv": "banking_data/test.csv",
    "categories.json": "banking_data/categories.json",
    "LICENSE": "LICENSE",
    "README.md": "README.md",
}
SPLITS = ("train", "validation", "test")
BASE_PARTITIONS = ("dirichlet_alpha_0.1", "dirichlet_alpha_0.2")
EXTRA_PARTITIONS = ("dirichlet_alpha_0.01",)
SUBSET_PARTITIONS = BASE_PARTITIONS + EXTRA_PARTITIONS
FULL_PARTITIONS = tuple(name + "_full" for name in BASE_PARTITIONS)
SUPPORTED_PARTITIONS = SUBSET_PARTITIONS + FULL_PARTITIONS


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path, fieldnames, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def ensure_raw(raw_dir):
    """Download a pinned upstream snapshot, or verify an existing snapshot."""
    raw_dir = Path(raw_dir)
    manifest_path = raw_dir / "download-manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest["files"]:
            path = raw_dir / item["path"]
            if not path.is_file() or sha256(path) != item["sha256"]:
                raise ValueError("Raw download checksum mismatch: " + str(path))
        return
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for name, source_path in RAW_FILES.items():
        path = raw_dir / name
        with urlopen(SOURCE_BASE + source_path, timeout=120) as response:
            content = response.read()
        temporary = path.with_name(path.name + ".part")
        temporary.write_bytes(content)
        temporary.replace(path)
        item = {
            "path": name, "source_path": source_path,
            "url": SOURCE_BASE + source_path, "revision": REVISION,
            "sha256": sha256(path), "size_bytes": len(content),
        }
        if name.endswith(".csv"):
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            item.update(row_count=len(rows), label_count=len({row["category"] for row in rows}))
        files.append(item)
    write_json(manifest_path, {"revision": REVISION, "files": files})


def normalized_text(text):
    return unicodedata.normalize("NFKC", " ".join(text.split())).casefold()


def load_raw(raw_dir):
    raw_dir = Path(raw_dir)
    labels = json.loads((raw_dir / "categories.json").read_text(encoding="utf-8"))
    if len(labels) != 77 or len(set(labels)) != 77:
        raise ValueError("Expected 77 unique Banking77 labels")
    label_ids = {name: index for index, name in enumerate(labels)}
    result = {}
    for split, expected in (("train", 10003), ("test", 3080)):
        with (raw_dir / (split + ".csv")).open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected:
            raise ValueError("Unexpected official " + split + " row count")
        result[split] = [
            {"id": "{}:{:05d}".format(split, index), "text": row["text"],
             "label": label_ids[row["category"]], "label_name": row["category"],
             "source_split": split, "source_row": index}
            for index, row in enumerate(rows)
        ]
    return labels, result


def clean_training_records(raw):
    """Apply the same text cleaning before either subset or full preparation."""
    held_out = {normalized_text(record["text"]) for record in raw["test"]}
    training_labels = defaultdict(set)
    for record in raw["train"]:
        training_labels[normalized_text(record["text"])].add(record["label"])
    conflicts = {text for text, labels in training_labels.items() if len(labels) > 1}
    by_label = defaultdict(list)
    seen = set()
    excluded = []
    for record in raw["train"]:
        key = normalized_text(record["text"])
        reason = None
        if key in held_out:
            reason = "text_also_in_official_test"
        elif key in conflicts:
            reason = "conflicting_training_labels"
        elif key in seen:
            reason = "duplicate_training_text"
        if reason:
            excluded.append({"id": record["id"], "reason": reason})
        else:
            by_label[record["label"]].append(record)
            seen.add(key)
    return by_label, excluded


def sample_pools(raw, seed):
    """Keep all test rows; reserve 10 validation and at most 50 training per class."""
    by_label, excluded = clean_training_records(raw)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    pools = {split: [] for split in SPLITS}
    eligible_counts = {}
    allocations = []
    for label in range(77):
        records = by_label[label]
        eligible_counts[str(label)] = len(records)
        if len(records) <= 10:
            raise ValueError("Too few leakage-free candidates for train/validation: label " + str(label))
        indices = rng.permutation(len(records))
        n_train = min(50, len(records) - 10)
        pools["train"].extend(records[index] for index in indices[:n_train])
        pools["validation"].extend(records[index] for index in indices[n_train:n_train + 10])
        allocations.append({"label": label, "label_name": records[0]["label_name"],
                            "eligible": len(records), "train": n_train, "validation": 10})
    pools["test"] = list(raw["test"])
    text_sets = {split: {normalized_text(record["text"]) for record in records}
                 for split, records in pools.items()}
    overlap = {left + "_" + right: len(text_sets[left] & text_sets[right])
               for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    if any(overlap.values()):
        raise ValueError("Normalized-text leakage in selected pools")
    return pools, {
        "normalization": "Unicode NFKC, collapse whitespace, casefold",
        "excluded_training_rows": excluded,
        "excluded_by_reason": dict(Counter(item["reason"] for item in excluded)),
        "eligible_training_rows_per_label": eligible_counts,
        "per_label_allocation": allocations,
        "sampling_rule": "Reserve 10 validation samples per class; sample up to 50 training samples without replacement",
        "selected_counts": {split: len(records) for split, records in pools.items()},
        "normalized_text_overlap_between_selected_splits": overlap,
        "test_rows_preserved": len(pools["test"]) == len(raw["test"]),
    }


def expand_pools(raw, reference_pools):
    """Use all eligible training rows while retaining the existing held-out IDs."""
    by_label, excluded = clean_training_records(raw)
    eligible = [record for label in range(77) for record in by_label[label]]
    by_id = {record["id"]: record for record in eligible}
    selected = reference_pools["train"] + reference_pools["validation"]
    if (len({record["id"] for record in selected}) != len(selected)
            or any(record != by_id.get(record["id"]) for record in selected)):
        raise ValueError("Reference train/validation must be distinct eligible raw records")
    validation = reference_pools["validation"]
    if Counter(record["label"] for record in validation) != {label: 10 for label in range(77)}:
        raise ValueError("Reference validation must retain ten samples per label")
    if reference_pools["test"] != raw["test"]:
        raise ValueError("Reference test must preserve the complete official test records")
    selected_ids = {record["id"] for record in selected}
    added = [record for record in eligible if record["id"] not in selected_ids]
    pools = {"train": list(reference_pools["train"]) + added,
             "validation": list(validation), "test": list(reference_pools["test"])}
    allocations = [{"label": label, "label_name": by_label[label][0]["label_name"],
                    "eligible": len(by_label[label]), "train": len(by_label[label]) - 10,
                    "validation": 10} for label in range(77)]
    audit = {
        "normalization": "Unicode NFKC, collapse whitespace, casefold",
        "excluded_training_rows": excluded,
        "excluded_by_reason": dict(Counter(item["reason"] for item in excluded)),
        "eligible_training_rows_per_label": {str(label): len(by_label[label]) for label in range(77)},
        "per_label_allocation": allocations,
        "sampling_rule": "Keep reference validation/test; append every remaining eligible training record",
        "reference_selected_counts": {split: len(reference_pools[split]) for split in SPLITS},
        "added_training_rows": len(added),
        "selected_counts": {split: len(pools[split]) for split in SPLITS},
        "test_rows_preserved": True,
    }
    return pools, audit


def integer_counts(probabilities, total):
    """Hamilton/largest-remainder allocation, stable ties by client ID."""
    targets = np.asarray(probabilities) * total
    counts = np.floor(targets).astype(int)
    remaining = int(total - counts.sum())
    order = np.argsort(-(targets - counts), kind="stable")
    counts[order[:remaining]] += 1
    if counts.sum() != total or (counts < 0).any():
        raise ValueError("Invalid integer allocation")
    return counts


def allocation_counts(probabilities, pools):
    result = {}
    for split in SPLITS:
        totals = Counter(record["label"] for record in pools[split])
        result[split] = np.stack([integer_counts(probabilities[label], totals[label])
                                  for label in range(77)], axis=0)
    return result


def make_partition(pools, seed, name):
    if name not in SUBSET_PARTITIONS:
        raise ValueError("Unknown partition: " + name)
    alpha = float(name.rsplit("_", 1)[1])
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    for attempt in range(1, 10001):
        probabilities = rng.dirichlet(np.full(10, alpha), size=77)
        counts = allocation_counts(probabilities, pools)
        if (counts["train"].sum(axis=0).min() >= 100
                and counts["test"].sum(axis=0).min() >= 100
                and counts["validation"].sum(axis=0).min() >= 1):
            break
    else:
        raise ValueError("Failed to meet client sample constraints in 10000 attempts")
    metadata = {"name": name, "seed": seed, "alpha": alpha, "attempts": attempt,
                "minimum_client_counts": {"train": 100, "validation": 1, "test": 100},
                "sampling_note": "Whole probability matrix rejection sampling, conditioned on minimum counts"}
    metadata.update(
        client_given_label_probabilities=probabilities.tolist(),
        integer_allocation="largest remainder; ties resolved by ascending client ID",
        counts_by_label_then_client={split: matrix.tolist() for split, matrix in counts.items()},
    )
    rng = np.random.default_rng(np.random.SeedSequence([seed, 4]))
    clients = [{split: [] for split in SPLITS} for _ in range(10)]
    for split in SPLITS:
        by_label = defaultdict(list)
        for record in pools[split]:
            by_label[record["label"]].append(record)
        for label in range(77):
            records = by_label[label]
            groups = defaultdict(list)
            for record in records:
                groups[normalized_text(record["text"])].append(record)
            # Official test has a normalized duplicate pair: retain both rows,
            # put the pair on one client, and preserve exact class/client counts.
            remaining = counts[split][label].copy()
            for group in groups.values():
                if len(group) > 1:
                    eligible = np.flatnonzero(remaining >= len(group))
                    if not len(eligible):
                        raise ValueError("Cannot co-locate duplicate group within prescribed class counts")
                    weights = remaining[eligible] / remaining[eligible].sum()
                    client = int(rng.choice(eligible, p=weights))
                    clients[client][split].extend(group)
                    remaining[client] -= len(group)
            singletons = [group[0] for group in groups.values() if len(group) == 1]
            shuffled = [singletons[index] for index in rng.permutation(len(singletons))]
            cursor = 0
            for client in range(10):
                size = int(remaining[client])
                clients[client][split].extend(shuffled[cursor:cursor + size])
                cursor += size
        for client in clients:
            records = client[split]
            client[split] = [records[index] for index in rng.permutation(len(records))]
    validate_clients(clients, pools, name)
    return clients, metadata


def make_full_partition(pools, seed, reference_clients, reference_metadata):
    """Append training rows using the original Dirichlet probabilities."""
    name = reference_metadata["name"]
    if name not in BASE_PARTITIONS or reference_metadata["seed"] != seed:
        raise ValueError("Full preparation requires a subset partition with the same seed")
    reference_pools = {split: [row for client in reference_clients for row in client[split]]
                       for split in SPLITS}
    validate_clients(reference_clients, reference_pools, name)
    probabilities = np.asarray(reference_metadata["client_given_label_probabilities"], dtype=float)
    if (probabilities.shape != (77, 10) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1)):
        raise ValueError("Invalid reference client probabilities")
    train_by_id = {record["id"]: record for record in pools["train"]}
    previous = reference_pools["train"]
    if any(record != train_by_id.get(record["id"]) for record in previous):
        raise ValueError("Full training pool must contain the unchanged subset training records")
    previous_ids = {record["id"] for record in previous}
    additional = defaultdict(list)
    for record in pools["train"]:
        if record["id"] not in previous_ids:
            additional[record["label"]].append(record)
    clients = [{split: list(client[split]) for split in SPLITS} for client in reference_clients]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 6]))
    for label in range(77):
        records = additional[label]
        shuffled = [records[index] for index in rng.permutation(len(records))]
        counts = integer_counts(probabilities[label], len(records))
        cursor = 0
        for client, count in zip(clients, counts):
            client["train"].extend(shuffled[cursor:cursor + count])
            cursor += count
    metadata = dict(reference_metadata)
    metadata.update(
        name=name + "_full", base_partition=name, full_training=True,
        reference_training_counts=[len(client["train"]) for client in reference_clients],
        additional_training_samples=sum(map(len, additional.values())),
        sampling_note="Reuse reference probabilities and assignments; no probability resampling",
        integer_allocation="Preserve subset quotas; largest remainder applied only to additional training rows",
        counts_by_label_then_client={split: np.stack([
            np.bincount([row["label"] for row in client[split]], minlength=77)
            for client in clients], axis=1).tolist() for split in SPLITS},
    )
    validate_clients(clients, pools, metadata["name"])
    return clients, metadata


def validate_clients(clients, pools, name):
    if name not in SUPPORTED_PARTITIONS:
        raise ValueError("Unknown partition: " + name)
    if len(clients) != 10:
        raise ValueError("Expected ten clients")
    for split in SPLITS:
        observed = [record for client in clients for record in client[split]]
        if len({record["id"] for record in pools[split]}) != len(pools[split]):
            raise ValueError("Duplicate sample IDs in " + split + " pool")
        if Counter(record["id"] for record in observed) != Counter(record["id"] for record in pools[split]):
            raise ValueError("Client partition does not exactly cover " + split)
        expected = {record["id"]: record for record in pools[split]}
        if any(record != expected[record["id"]] for record in observed):
            raise ValueError("Record content changed during client allocation")
        text_owners = defaultdict(set)
        for client_id, client in enumerate(clients):
            for record in client[split]:
                text_owners[normalized_text(record["text"])].add(client_id)
        if any(len(owners) != 1 for owners in text_owners.values()):
            raise ValueError("Normalized duplicate text assigned to multiple clients")
    if any(len(client["train"]) < 100 or len(client["test"]) < 100
           or not client["validation"] for client in clients):
        raise ValueError("Minimum client sample constraints violated")


def js_divergence(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left = left / left.sum()
    right = right / right.sum()
    middle = (left + right) / 2
    result = 0.0
    for values in (left, right):
        nonzero = values > 0
        result += 0.5 * float(np.sum(values[nonzero] * np.log2(values[nonzero] / middle[nonzero])))
    return max(0.0, min(1.0, result))


def partition_stats(clients, labels, metadata):
    display_name = "Dirichlet alpha=" + str(metadata["alpha"])
    if metadata["name"].endswith("_full"):
        display_name += " (full training pool)"
    stats = {"name": metadata["name"], "seed": metadata["seed"],
             "display_name": display_name,
             "alpha": metadata.get("alpha"), "label_names": labels, "clients": []}
    for client_id, client in enumerate(clients):
        item = {"client_id": client_id, "splits": {}}
        for split, records in client.items():
            counts = np.bincount([record["label"] for record in records], minlength=77)
            probabilities = counts[counts > 0] / counts.sum()
            entropy = float(-np.sum(probabilities * np.log2(probabilities)))
            item["splits"][split] = {
                "n_samples": len(records), "label_counts": counts.tolist(),
                "n_labels": int((counts > 0).sum()), "label_entropy_bits": entropy,
                "normalized_label_entropy": entropy / float(np.log2(77)), "effective_labels": 2 ** entropy,
                "mean_words": float(np.mean([len(record["text"].split()) for record in records])),
                "mean_chars": float(np.mean([len(record["text"]) for record in records])),
            }
        item["split_js_divergence"] = {
            left + "_" + right: js_divergence(item["splits"][left]["label_counts"],
                                              item["splits"][right]["label_counts"])
            for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
        }
        stats["clients"].append(item)
    stats["pairwise_train_js_divergence"] = [
        [js_divergence(left["splits"]["train"]["label_counts"], right["splits"]["train"]["label_counts"])
         for right in stats["clients"]] for left in stats["clients"]
    ]
    matrix = np.asarray(stats["pairwise_train_js_divergence"])
    stats["summary"] = {
        "mean_pairwise_train_js_divergence": float(matrix[np.triu_indices(10, k=1)].mean()),
        "mean_train_validation_js_divergence": float(np.mean([
            client["split_js_divergence"]["train_validation"] for client in stats["clients"]])),
        "mean_train_test_js_divergence": float(np.mean([
            client["split_js_divergence"]["train_test"] for client in stats["clients"]])),
    }
    return stats


def save_partition(path, clients, labels, metadata):
    path = Path(path)
    stats = partition_stats(clients, labels, metadata)
    write_json(path / "partition.json", metadata)
    write_json(path / "stats.json", stats)
    client_rows, count_rows = [], []
    for client, item in zip(clients, stats["clients"]):
        for split, records in client.items():
            write_jsonl(path / "clients" / ("client_{:02d}".format(item["client_id"])) / (split + ".jsonl"), records)
            split_stats = dict(item["splits"][split])
            label_counts = split_stats.pop("label_counts")
            client_rows.append(dict(client_id=item["client_id"], split=split, **split_stats))
            count_rows.extend({"client_id": item["client_id"], "split": split, "label": label,
                               "label_name": labels[label], "count": count,
                               "frequency": count / len(records)} for label, count in enumerate(label_counts))
    write_csv(path / "client_stats.csv", list(client_rows[0]), client_rows)
    write_csv(path / "label_counts.csv", list(count_rows[0]), count_rows)
    return stats


def verify_prepared(output_dir, raw_dir=None):
    """Verify persisted hashes and independently reload all client assignments."""
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    partitions = list(manifest["partitions"])
    if any(name not in SUPPORTED_PARTITIONS for name in partitions):
        raise ValueError("Prepared data uses an unsupported partition; regenerate it with prepare_data.py")
    for item in manifest["files"]:
        if sha256(output_dir / item["path"]) != item["sha256"]:
            raise ValueError("Prepared file checksum mismatch: " + item["path"])
    for name in EXTRA_PARTITIONS:
        directory = output_dir / name
        if name in partitions or not directory.exists():
            continue
        extra = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if extra["name"] != name or extra["seed"] != manifest["seed"]:
            raise ValueError("Extra partition name/seed mismatch: " + name)
        for path, expected_hash in extra["reference_files"].items():
            if sha256(output_dir / path) != expected_hash:
                raise ValueError("Extra partition reference changed: " + path)
        for item in extra["files"]:
            if sha256(directory / item["path"]) != item["sha256"]:
                raise ValueError("Extra partition checksum mismatch: " + name + "/" + item["path"])
        partitions.append(name)
    pools = {split: [json.loads(line) for line in (output_dir / "pools" / (split + ".jsonl")).read_text(encoding="utf-8").splitlines()]
             for split in SPLITS}
    labels = json.loads((output_dir / "labels.json").read_text(encoding="utf-8"))
    if {split: len(records) for split, records in pools.items()} != manifest["sample_counts"]:
        raise ValueError("Pool counts disagree with the preparation manifest")
    if raw_dir is not None:
        raw_dir = Path(raw_dir)
        for name, expected_hash in manifest["raw_sha256"].items():
            if sha256(raw_dir / name) != expected_hash:
                raise ValueError("Raw file differs from the snapshot used for preparation: " + name)
        raw_labels, raw = load_raw(raw_dir)
        if labels != raw_labels or pools["test"] != raw["test"]:
            raise ValueError("Prepared labels/test pool differ from the complete official source")
        if manifest["sampling"].get("full_training"):
            by_label, _ = clean_training_records(raw)
            eligible = {row["id"]: row for rows in by_label.values() for row in rows}
            selected = pools["train"] + pools["validation"]
            if len(selected) != len(eligible) or {row["id"]: row for row in selected} != eligible:
                raise ValueError("Full train/validation must exactly cover all eligible raw training rows")
    for name, value in manifest.get("reference_evaluation_sha256", {}).items():
        if sha256(output_dir / name) != value:
            raise ValueError("Full data changed reference validation/test records: " + name)
    for name, reference in manifest.get("reference_training_prefixes", {}).items():
        rows = (output_dir / name).read_bytes().splitlines(keepends=True)
        prefix = b"".join(rows[:reference["rows"]])
        if hashlib.sha256(prefix).hexdigest() != reference["sha256"]:
            raise ValueError("Full data changed the original client training prefix: " + name)
    for name in partitions:
        clients = [{split: [json.loads(line) for line in (
            output_dir / name / "clients" / ("client_{:02d}".format(client)) / (split + ".jsonl")
        ).read_text(encoding="utf-8").splitlines()] for split in SPLITS} for client in range(10)]
        validate_clients(clients, pools, name)
        metadata = json.loads((output_dir / name / "partition.json").read_text(encoding="utf-8"))
        for split in SPLITS:
            observed = np.stack([np.bincount([record["label"] for record in client[split]], minlength=77)
                                 for client in clients], axis=1)
            quotas = np.asarray(metadata["counts_by_label_then_client"][split])
            if not np.array_equal(observed, quotas):
                raise ValueError("Client records disagree with stored allocation quotas")
        expected = partition_stats(clients, labels, metadata)
        saved = json.loads((output_dir / name / "stats.json").read_text(encoding="utf-8"))
        if expected != saved:
            raise ValueError("Saved statistics do not match client records")
    normalized = {split: {normalized_text(record["text"]) for record in pools[split]} for split in SPLITS}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if normalized[left] & normalized[right]:
            raise ValueError("Text leakage found on persisted splits")
    checks = ["file checksums", "sample coverage and uniqueness", "client minimum sizes",
              "persisted statistics", "normalized-text separation",
              "normalized duplicates assigned to a single client", "allocation quotas match client records"]
    if raw_dir is not None:
        checks.extend(["raw snapshot checksums", "complete official test records preserved verbatim"])
        if manifest["sampling"].get("full_training"):
            checks.append("all eligible training rows used outside the fixed validation pool")
    if manifest.get("reference_evaluation_sha256"):
        checks.extend(["reference validation/test assignments unchanged", "original client training prefixes unchanged"])
    return {"status": "passed", "partitions_verified": len(partitions),
            "clients_per_partition": 10, "pool_counts": {split: len(records) for split, records in pools.items()},
            "checks": checks}
