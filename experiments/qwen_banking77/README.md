# FC-MOO-Qwen_fair

This project contains the language model experiments for FC-MOO. It uses LoRA to fine-tune Qwen2.5-3B-Instruct for federated intent classification on Banking77 with ten clients.

See the main [FC-MOO repository](https://github.com/Zhg9300/FC-MOO) for image classification experiments on FashionMNIST, CIFAR-10, and CIFAR-100.

## Quick Start

Run the commands below from the directory containing this README.

### 1. Install Dependencies

The reference environment uses Linux on ARM64, Python 3.9.5, PyTorch 2.1.0, `torch_npu` 2.1.0.post8, and CANN 8.3.RC1. Training on Ascend requires an NPU with BF16 support. Follow the installation instructions for your device and driver to install compatible versions of PyTorch, `torch_npu`, and CANN, then source CANN's `set_env.sh` before training. The reference environment also uses `setuptools` 69.5.1, `decorator` 5.1.1, and SciPy 1.13.1.

The project uses NumPy, Transformers, PEFT, Accelerate, Safetensors, and PyYAML. Their versions are listed in [`requirements.txt`](requirements.txt). This command also installs the Hugging Face CLI for downloading the model:

```bash
python -m pip install -r requirements.txt "huggingface_hub==0.36.2"
```

Data preparation alone requires only the dependencies in `requirements-data.txt`. Install them with `python -m pip install -r requirements-data.txt`; PyTorch and an NPU are not required.

### 2. Prepare the Data and Model

`prepare_data.py` downloads a pinned revision of [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data) and generates two partitions by default: `dirichlet_alpha_0.1` and `dirichlet_alpha_0.2`, each with ten clients. Then download the Qwen2.5-3B-Instruct weights and tokenizer at the revision shown below:

```bash
python prepare_data.py
python prepare_data.py --verify-only

hf download Qwen/Qwen2.5-3B-Instruct \
  --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 \
  --local-dir Qwen2.5-3B-Instruct
```

If you already prepared the data with the current code, run `python prepare_data.py --verify-only`. If you prepared it with an earlier version, reuse the raw data and regenerate the partitions in an empty directory. Once the default partitions are ready, run `python prepare_data.py --partition dirichlet_alpha_0.01` to add the α = 0.01 partition.

### 3. Run a Smoke Test

Run three training rounds with the prepared model and data. The smoke test evaluates up to eight samples per client on each enabled evaluation split:

```bash
python run.py --method fcmoo --smoke
```

Each run writes to `outputs/<run>/` by default and prints the actual path as `RUN_DIRECTORY`. In that directory, `result.json` contains the final evaluation and `metrics.jsonl` logs training and evaluation metrics by round.

## Default Experiment and Usage

[`configs/default.yaml`](configs/default.yaml) defines the default experiment:

| Setting | Default |
| --- | --- |
| Method | `fcmoo` |
| Data partition | Banking77, 10 clients, Dirichlet α = 0.1 |
| Data seed / training seed | `2026` / `2026` |
| Training rounds | `500` |
| Evaluation splits | Training and validation |
| Training device | `npu:0` |

Train with the default configuration or use `--partition` to select α = 0.2:

```bash
python run.py --method fcmoo
python run.py --method fcmoo --partition dirichlet_alpha_0.2
```

The `--method` option also accepts `fedavg`, `qffl`, `fedmgda`, and `fairmoo`. Use `--device` to select a device and `--rounds` to override the number of rounds. To use a model or data stored elsewhere, copy the default configuration, update `model.model_path`, `data.raw_dir`, and `data.prepared_root` as needed, then pass the new file with `--config`. Run `python run.py --help` for all options.

## Unit Tests

The unit tests run on CPU with PyTorch and the dependencies in `requirements.txt`. No model weights are needed. Tests that require the Banking77 partitions are skipped if the data has not been prepared.

```bash
python -m unittest discover -s tests -v
```

## License

Banking77 is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Qwen2.5-3B-Instruct is subject to the [Qwen Research License Agreement](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE).
