#!/usr/bin/env python3
"""Train one method on a Banking77 partition."""

import argparse
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import uuid

import yaml

from fl_exp.data import SUPPORTED_PARTITIONS


ROOT = Path(__file__).resolve().parent
METHODS = ("fedavg", "qffl", "fedmgda", "fairmoo", "fcmoo")


def validate(config):
    training = config["training"]
    head_scale = float(config["model"].get("head_init_scale", 1.0))
    if not math.isfinite(head_scale) or head_scale <= 0:
        raise ValueError("head_init_scale must be finite and positive")
    for key in ("rounds", "batch_size", "micro_batch_size", "eval_batch_size", "eval_every", "checkpoint_every", "lr", "lr_offset"):
        if training[key] <= 0:
            raise ValueError(key + " must be positive")
    if config["seed"] < 0 or config["model"]["max_length"] < 1 or training["lr_power"] < 0:
        raise ValueError("Invalid seed, maximum length, or learning-rate exponent")
    data_seed = config["data"].get("seed", config["seed"])
    if not isinstance(data_seed, int) or data_seed < 0:
        raise ValueError("data.seed must be a nonnegative integer")
    cap = training["eval_max_samples_per_client"]
    if cap is not None and cap < 1:
        raise ValueError("Evaluation sample cap must be positive or null")
    splits = training.get("evaluation_splits", ["train", "validation", "test"])
    periodic = training.get("periodic_eval_splits", ["validation"])
    if (len(splits) != len(set(splits)) or not {"train", "validation"}.issubset(splits)
            or not set(splits).issubset({"train", "validation", "test"})
            or not set(periodic).issubset(splits)):
        raise ValueError("Evaluation splits must include train/validation; periodic splits must be enabled")
    for section, key in (("model", "model_path"), ("data", "prepared_root"), ("data", "raw_dir")):
        path = Path(config[section][key]).expanduser()
        config[section][key] = str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--partition", choices=SUPPORTED_PARTITIONS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--device", help="Logical device, e.g. npu:0 or cpu")
    parser.add_argument("--eval-max-samples-per-client", type=int)
    parser.add_argument("--smoke", action="store_true", help="Three rounds, at most 8 evaluation samples per client")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="Checkpoint directory; --rounds is the desired total")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.smoke:
        config["smoke_test"] = True
        config["training"].update(rounds=3, eval_every=3, checkpoint_every=3,
                                  eval_max_samples_per_client=8, evaluate_initial=True)
    for key in ("method", "partition", "seed"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    for key in ("rounds", "lr", "batch_size", "micro_batch_size", "device", "eval_max_samples_per_client"):
        if getattr(args, key) is not None:
            config["training"][key] = getattr(args, key)
    validate(config)
    if config["method"] not in METHODS or config["partition"] not in SUPPORTED_PARTITIONS:
        parser.error("Unknown configured method or partition")
    tag = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    directory = (args.output_dir or ROOT / "outputs" / (
        f"{tag}_{config['partition']}_{config['method']}_seed{config['seed']}_{uuid.uuid4().hex[:6]}"
    )).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print("RUN_DIRECTORY=" + str(directory), flush=True)
    resume = args.resume.resolve() if args.resume else None
    try:
        from fl_exp.trainer import train
        train(config, directory, resume)
    except BaseException as error:
        (directory / "failure.json").write_text(json.dumps({"status": "failed", "error": str(error)}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
