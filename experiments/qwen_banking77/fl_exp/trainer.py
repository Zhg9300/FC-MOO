"""One global model, ten serial client gradients, and one update per round."""

import contextlib
import hashlib
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .algorithms import FederatedAlgorithm
from .data import sha256, verify_prepared, write_json
from .metrics import summarize_clients
from .model import build_model, trainable_named_parameters


def setup_device(name, seed):
    if name.startswith("npu"):
        import torch_npu  # noqa: F401
    device = torch.device(name)
    torch.set_num_threads(4)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type != "cpu":
        backend = getattr(torch, device.type)
        backend.set_device(device)
        backend.manual_seed_all(seed)
        if not backend.is_bf16_supported():
            raise RuntimeError("This configuration requires BF16 support")
    return device


def autocast(device):
    return (contextlib.nullcontext() if device.type == "cpu" else
            torch.autocast(device_type=device.type, dtype=torch.bfloat16))


def synchronize(device):
    if device.type != "cpu":
        getattr(torch, device.type).synchronize()


def parameter_vector(named, gradients=False):
    values = []
    for name, parameter in named:
        value = parameter.grad if gradients else parameter.detach()
        if value is None:
            raise RuntimeError("Missing gradient: " + name)
        values.append(value.reshape(-1))
    return torch.cat(values).detach().float().cpu()


def frozen_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(str((tuple(parameter.shape), parameter.dtype)).encode())
            raw = parameter.detach().cpu().contiguous().view(torch.uint8).numpy()
            digest.update(memoryview(raw))
    return digest.hexdigest()


class ClientData:
    """CPU token tensors and independent per-client minibatch RNG streams."""

    def __init__(self, config, tokenizer):
        seed = config["seed"]
        self.data_seed = config["data"].get("seed", seed)
        prepared = Path(config["data"]["prepared_root"]) / ("seed_" + str(self.data_seed))
        verify_prepared(prepared, Path(config["data"]["raw_dir"]))
        self.manifest_sha256 = sha256(prepared / "manifest.json")
        manifest = json.loads((prepared / "manifest.json").read_text())
        if manifest["seed"] != self.data_seed or manifest["n_clients"] != 10 or manifest["n_labels"] != config["model"]["num_labels"]:
            raise ValueError("Prepared data does not match experiment configuration")
        self.clients, self.eval_indices = [], []
        self.dynamic_padding = bool(config["training"].get("dynamic_padding", False))
        self.generators = [np.random.default_rng(np.random.SeedSequence([seed, 100, client])) for client in range(10)]
        cap = config["training"]["eval_max_samples_per_client"]
        evaluation_splits = config["training"].get("evaluation_splits", ["train", "validation", "test"])
        for client in range(10):
            client_data, evaluation = {}, {}
            for split_id, split in enumerate(("train", "validation", "test")):
                if split not in evaluation_splits:
                    continue
                path = prepared / config["partition"] / "clients" / ("client_{:02d}".format(client)) / (split + ".jsonl")
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                encoded = tokenizer([row["text"] for row in rows], padding="max_length", truncation=True,
                                    max_length=config["model"]["max_length"], return_tensors="pt")
                tensors = {key: encoded[key] for key in ("input_ids", "attention_mask")}
                tensors["labels"] = torch.tensor([row["label"] for row in rows], dtype=torch.long)
                client_data[split] = {"tensors": tensors, "ids": [row["id"] for row in rows]}
                n = len(rows)
                rng = np.random.default_rng(np.random.SeedSequence([self.data_seed, 200, client, split_id]))
                evaluation[split] = (np.sort(rng.choice(n, size=cap, replace=False)).tolist()
                                     if cap is not None and n > cap else list(range(n)))
            self.clients.append(client_data)
            self.eval_indices.append(evaluation)

    def batch(self, client, split, indices, device):
        indices = torch.tensor(indices, dtype=torch.long)
        batch = {key: values.index_select(0, indices)
                 for key, values in self.clients[client][split]["tensors"].items()}
        if self.dynamic_padding:
            valid_length = max(1, int(batch["attention_mask"].sum(dim=1).max().item()))
            length = min(batch["input_ids"].shape[1], ((valid_length + 15) // 16) * 16)
            for key in ("input_ids", "attention_mask"):
                batch[key] = batch[key][:, :length].contiguous()
        return {key: values.to(device) for key, values in batch.items()}

    def sample(self, client, size):
        n = len(self.clients[client]["train"]["ids"])
        if size > n:
            raise ValueError("Effective minibatch exceeds client training set")
        return self.generators[client].choice(n, size=size, replace=False).tolist()

    def evaluation_manifest(self):
        return [{"client_id": client,
                 "sample_ids": {split: [self.clients[client][split]["ids"][i] for i in indices]
                                for split, indices in selection.items()}}
                for client, selection in enumerate(self.eval_indices)]


@torch.no_grad()
def evaluate(model, data, split, device, batch_size):
    was_training = model.training
    model.eval()
    clients = []
    for client in range(10):
        selected = data.eval_indices[client][split]
        loss_sum, correct = 0.0, 0
        for start in range(0, len(selected), batch_size):
            batch = data.batch(client, split, selected[start:start + batch_size], device)
            with autocast(device):
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            loss_sum += F.cross_entropy(logits.float(), batch["labels"], reduction="sum").item()
            correct += (logits.argmax(dim=-1) == batch["labels"]).sum().item()
        clients.append({"client_id": client, "n_samples": len(selected),
                        "n_available": len(data.clients[client][split]["ids"]),
                        "loss": loss_sum / len(selected), "accuracy": correct / len(selected)})
    model.train(was_training)
    scope = "full" if all(row["n_samples"] == row["n_available"] for row in clients) else "subset"
    return {"metrics": summarize_clients(clients), "clients": clients, "evaluation_scope": scope}


def save_checkpoint(path, model, named, algorithm, data, config, round_index, frozen, device):
    path.mkdir(parents=True, exist_ok=True)
    state = {
        "round": round_index, "config": config, "frozen_sha256": frozen,
        "prepared_manifest_sha256": data.manifest_sha256,
        "trainables": {name: parameter.detach().cpu().clone() for name, parameter in named},
        "algorithm": algorithm.state_dict(),
        "samplers": [rng.bit_generator.state for rng in data.generators],
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "device_rng": None if device.type == "cpu" else getattr(torch, device.type).get_rng_state(device),
    }
    temporary = path / "state.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(path / "state.pt")
    model.save_pretrained(path / "adapter", safe_serialization=True, save_embedding_layers=False)


def restore_checkpoint(path, model, named, algorithm, data, config, frozen, device):
    state = torch.load(Path(path) / "state.pt", map_location="cpu")
    previous = state["config"]
    for key in ("seed", "method", "partition", "model", "data", "algorithms", "smoke_test"):
        if previous[key] != config[key]:
            raise ValueError("Resume configuration differs: " + key)
    for key, value in previous["training"].items():
        if key not in ("rounds", "eval_every", "checkpoint_every") and config["training"][key] != value:
            raise ValueError("Resume training configuration differs: " + key)
    if state["frozen_sha256"] != frozen or state["prepared_manifest_sha256"] != data.manifest_sha256:
        raise ValueError("Resume backbone or prepared data changed")
    if list(state["trainables"]) != [name for name, _ in named]:
        raise ValueError("Checkpoint trainable parameter names differ")
    with torch.no_grad():
        for name, parameter in named:
            saved = state["trainables"][name]
            if saved.dtype != torch.float32 or saved.shape != parameter.shape:
                raise ValueError("Checkpoint trainable dtype/shape mismatch")
            parameter.copy_(saved.to(device))
            if not torch.equal(parameter.detach().cpu(), saved):
                raise RuntimeError("Checkpoint restore was not exact")
    algorithm.load_state_dict(state["algorithm"])
    for rng, saved in zip(data.generators, state["samplers"]):
        rng.bit_generator.state = saved
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if device.type != "cpu":
        getattr(torch, device.type).set_rng_state(state["device_rng"], device)
    return state["round"]


def train(config, output_dir, resume=None):
    output_dir = Path(output_dir)
    log = logging.getLogger(str(output_dir))
    log.setLevel(logging.INFO)
    log.propagate = False
    handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(handler)

    def record(value):
        text = json.dumps(value, ensure_ascii=False, allow_nan=False)
        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
        log.info(text)
        print(text, flush=True)

    begin = time.monotonic()
    cfg = config["training"]
    device = setup_device(cfg["device"], config["seed"])
    log.info("Loading local Qwen model and prepared data")
    print("Loading local Qwen classification model...", flush=True)
    model, tokenizer = build_model(config["model"], device)
    named = trainable_named_parameters(model)
    data = ClientData(config, tokenizer)
    write_json(output_dir / "evaluation_samples.json", data.evaluation_manifest())
    algorithm = FederatedAlgorithm(config["method"], config["algorithms"], n_clients=10)
    frozen_before = frozen_hash(model)
    start_round = 0 if resume is None else restore_checkpoint(
        resume, model, named, algorithm, data, config, frozen_before, device)
    if start_round >= cfg["rounds"]:
        raise ValueError("Requested total rounds must exceed the checkpoint round")
    initial = parameter_vector(named)
    lora_mask = torch.cat([torch.full((parameter.numel(),), "lora_" in name, dtype=torch.bool)
                           for name, parameter in named])
    initial_evaluation = {}
    if cfg["evaluate_initial"]:
        for split in ("train", "validation"):
            evaluated = evaluate(model, data, split, device, cfg["eval_batch_size"])
            initial_evaluation[split] = evaluated
            record({"event": "evaluation", "round": start_round, "split": split, **evaluated})
    rounds = []
    for round_index in range(start_round + 1, cfg["rounds"] + 1):
        synchronize(device)
        round_begin = time.monotonic()
        if device.type != "cpu":
            getattr(torch, device.type).reset_peak_memory_stats(device)
        model.train()
        losses, gradients = [], []
        # No parameter update occurs until all ten client gradients are collected.
        for client in range(10):
            selected = data.sample(client, cfg["batch_size"])
            model.zero_grad(set_to_none=True)
            loss_value = 0.0
            for start in range(0, len(selected), cfg["micro_batch_size"]):
                indices = selected[start:start + cfg["micro_batch_size"]]
                batch = data.batch(client, "train", indices, device)
                with autocast(device):
                    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
                # Keep client losses and their gradients on the same FP32 CE
                # definition used in evaluation, including under NPU autocast.
                loss = F.cross_entropy(logits.float(), batch["labels"])
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Nonfinite client loss")
                weight = len(indices) / len(selected)
                (loss * weight).backward()
                loss_value += loss.item() * weight
            gradient = parameter_vector(named, gradients=True)
            if not torch.isfinite(gradient).all().item():
                raise FloatingPointError("Nonfinite client gradient")
            losses.append(loss_value)
            gradients.append(gradient)
        model.zero_grad(set_to_none=True)
        lr = cfg["lr"] * (1 + (round_index - 1) / cfg["lr_offset"]) ** (-cfg["lr_power"])
        direction, diagnostics = algorithm.direction(np.asarray(losses), torch.stack(gradients), round_index, lr)
        if not torch.isfinite(direction).all().item():
            raise FloatingPointError("Nonfinite global direction")
        before = parameter_vector(named)
        on_device = direction.to(device)
        offset = 0
        with torch.no_grad():
            for _, parameter in named:
                size = parameter.numel()
                parameter.add_(on_device[offset:offset + size].view_as(parameter), alpha=-lr)
                offset += size
        after = parameter_vector(named)
        update = after - before
        if not torch.isfinite(after).all().item() or not torch.allclose(update, -lr * direction, atol=2e-7, rtol=2e-5):
            raise FloatingPointError("Global SGD update is nonfinite or inconsistent")
        if any(parameter.grad is not None for parameter in model.parameters() if not parameter.requires_grad):
            raise RuntimeError("Frozen backbone has gradients")
        synchronize(device)
        row = {"event": "round", "round": round_index, "lr": lr,
               "client_losses": losses, "client_grad_norms": [float(gradient.norm()) for gradient in gradients],
               "sampled_loss_variance": float(np.var(losses)), "direction_norm": float(direction.norm()),
               "update_norm": float(update.norm()), "lora_update_norm": float(update[lora_mask].norm()),
               "head_update_norm": float(update[~lora_mask].norm()), "diagnostics": diagnostics,
               "elapsed_seconds": time.monotonic() - round_begin,
               "peak_allocated_bytes": 0 if device.type == "cpu" else getattr(torch, device.type).max_memory_allocated(device)}
        rounds.append(row)
        record(row)
        if round_index % cfg["eval_every"] == 0 and round_index < cfg["rounds"]:
            for split in cfg.get("periodic_eval_splits", ["validation"]):
                record({"event": "evaluation", "round": round_index, "split": split,
                        **evaluate(model, data, split, device, cfg["eval_batch_size"])})
        if round_index % cfg["checkpoint_every"] == 0 or round_index == cfg["rounds"]:
            save_checkpoint(output_dir / "checkpoint", model, named, algorithm, data, config,
                            round_index, frozen_before, device)
    final = {}
    for split in cfg.get("evaluation_splits", ["train", "validation", "test"]):
        final[split] = evaluate(model, data, split, device, cfg["eval_batch_size"])
        record({"event": "evaluation", "round": cfg["rounds"], "split": split, **final[split]})
    final["train"]["constraint_violation"] = max(final["train"]["metrics"]["loss_variance"] - config["algorithms"]["delta"], 0)
    final["train"]["constraint_violation_is_estimate"] = final["train"]["evaluation_scope"] != "full"
    frozen_after = frozen_hash(model)
    change = parameter_vector(named) - initial
    verification = {
        "frozen_weights_unchanged": frozen_before == frozen_after,
        "frozen_sha256_before": frozen_before, "frozen_sha256_after": frozen_after,
        "lora_changed": bool(change[lora_mask].norm() > 0),
        "classification_head_changed": bool(change[~lora_mask].norm() > 0),
        "finite_losses_gradients_parameters": True, "global_sgd_updates_checked": True,
        "cross_entropy_dtype": "float32",
        "trainable_parameter_count": sum(parameter.numel() for _, parameter in named),
        "trainable_parameter_names": [name for name, _ in named],
        "prepared_manifest_sha256": data.manifest_sha256,
        "resumed_from_round": start_round,
    }
    if not all(verification[key] for key in ("frozen_weights_unchanged", "lora_changed", "classification_head_changed")):
        raise RuntimeError("Frozen/trainable parameter verification failed")
    result = {"status": "completed", "method": config["method"], "partition": config["partition"],
              "seed": config["seed"], "data_seed": data.data_seed,
              "rounds": cfg["rounds"], "smoke_test": config["smoke_test"],
              "config": config, "initial": initial_evaluation, "final": final, "verification": verification,
              "timing": {"total_seconds": time.monotonic() - begin,
                         "training_seconds": sum(row["elapsed_seconds"] for row in rounds),
                         "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in rounds)}}
    write_json(output_dir / "result.json", result)
    log.info("Completed: %s", output_dir)
    handler.close()
    log.removeHandler(handler)
    return result
