# FC-MOO

A federated learning research framework for fairness-constrained multi-objective optimization and comparisons with existing federated learning methods.

## Experiments

- **Image classification** (FashionMNIST, CIFAR-10, CIFAR-100): see Quick Start below.
- **Federated intent classification** (Qwen2.5-3B-Instruct, Banking77): see the [Qwen / Banking77 experiments](experiments/qwen_banking77/README.md).

## Requirements

```text
numpy==1.25.1
pytorch==2.0.1
torchvision==0.15.2
cvxopt==1.3.2
```

## Quick Start

Run FC-MOO-sto with a CNN on CIFAR-10:

```bash
python run.py --seed 1 --device 0 --module CNN \
    --algorithm FC-MOO-sto --dataloader DataLoader_cifar10_pat \
    --N 10 --B 16 --NC 1 --R 4001 --E 1 \
    --lr 0.16 --decay 0.999 --momentum 0.9 \
    --sgd_step True --test_interval 200 --prefer 0.0001 --rho 5
```

Use `--algorithm FC-MOO` for the deterministic variant and `--device -1` for CPU.
Datasets are downloaded to `components/data/`; prepared client data is cached in
`components/pool/`.

Display all command-line options:

```bash
python run.py --help
```

## Supported Components

**Algorithms:** `FC-MOO`, `FC-MOO-sto`, `FairMOO`, `FedAvg`, `qFedAvg`, `AFL`,
`DRFL`, `FedMGDA_plus`, `FedMDFG`, `FedLF`, and `AdaFed`.

**Models:** `CNN`, `MLP`, `ResNet18`, `ResNet50`.
Use MLP for FashionMNIST and CNN or ResNet for CIFAR-10/100.
ResNet models accept `--pretrained True` for ImageNet initialization.

| Dataset | Pathological partition | Dirichlet partition |
| --- | --- | --- |
| FashionMNIST | `DataLoader_fashion_pat` | `DataLoader_fashion_dir` |
| CIFAR-10 | `DataLoader_cifar10_pat` | `DataLoader_cifar10_dir` |
| CIFAR-100 | `DataLoader_cifar100_pat`, `DataLoader_cifar100_pat1` | `DataLoader_cifar100_dir` |

## Key Arguments

| Argument | Meaning |
| --- | --- |
| `--prefer`, `--rho` | FC-MOO fairness threshold and constraint coefficient |
| `--N`, `--C` | Number of clients and participation fraction |
| `--R`, `--E` | Communication rounds and local epochs |
| `--B`, `--lr`, `--decay` | Batch size, learning rate, and learning rate decay |
| `--momentum`, `--sgd_step` | SGD momentum and single-step local training |
| `--NC`, `--Diralpha` | Classes per client and Dirichlet concentration |
| `--test_interval` | Evaluation interval in communication rounds |
