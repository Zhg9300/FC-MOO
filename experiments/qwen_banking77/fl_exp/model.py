"""Frozen local Qwen backbone with a shared LoRA adapter and classifier."""

import math
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def trainable_named_parameters(model):
    """Stable parameter order used by gradient vectors and checkpoints."""
    return [(name, parameter) for name, parameter in model.named_parameters()
            if parameter.requires_grad]


def build_model(config, device):
    """Load local weights; keep the global adapter and classification head FP32."""
    head_scale = float(config.get("head_init_scale", 1.0))
    if not math.isfinite(head_scale) or head_scale <= 0:
        raise ValueError("head_init_scale must be finite and positive")
    device = torch.device(device)
    path = Path(config["model_path"])
    if not path.is_dir():
        raise FileNotFoundError("Local model directory does not exist: " + str(path))
    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=False, use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define a padding or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    # transformers 4.41 Qwen pools just before the first pad token.
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    base = AutoModelForSequenceClassification.from_pretrained(
        path, num_labels=int(config.get("num_labels", 77)),
        torch_dtype=dtype, local_files_only=True, trust_remote_code=False,
        low_cpu_mem_usage=True, attn_implementation="eager",
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.problem_type = "single_label_classification"
    base.config.use_cache = False
    adapter = LoraConfig(
        task_type=TaskType.SEQ_CLS, bias="none",
        r=int(config.get("lora_r", 8)),
        lora_alpha=int(config.get("lora_alpha", 16)),
        lora_dropout=float(config.get("lora_dropout", 0.0)),
        target_modules=list(config.get("target_modules", ["q_proj", "v_proj"])),
        modules_to_save=["score"],
    )
    model = get_peft_model(base, adapter)
    for _, parameter in trainable_named_parameters(model):
        parameter.data = parameter.data.float()
    named = trainable_named_parameters(model)
    if not any("lora_" in name for name, _ in named):
        raise RuntimeError("No trainable LoRA parameters")
    if not any("score" in name for name, _ in named):
        raise RuntimeError("Shared classification head must be trainable")
    if any("lora_" not in name and "score" not in name for name, _ in named):
        raise RuntimeError("Unexpected trainable backbone parameters")
    heads = [parameter for name, parameter in named if "score" in name and name.endswith(".weight")]
    if len(heads) != 1:
        raise RuntimeError("Expected one active shared classification head")
    # Rescale the active FP32 head without drawing new random numbers or
    # modifying PEFT's frozen original_module copy. Scale 1 preserves old runs.
    if head_scale != 1.0:
        with torch.no_grad():
            heads[0].mul_(head_scale)
        if not torch.isfinite(heads[0]).all():
            raise ValueError("head_init_scale produced nonfinite classifier weights")
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        model.enable_input_require_grads()
    model.to(device)
    return model, tokenizer
