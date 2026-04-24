# Toy Continuous Learning on Split-CIFAR10 (Train During Inference)

This repository contains a lightweight continuous learning pipeline built from scratch for local GPU training (e.g., RTX 5080).

## What it does

- Streams Split-CIFAR10 samples one by one (class-incremental tasks).
- Runs inference first (prequential evaluation).
- Receives labels with delay (`--delay-steps`).
- Performs online updates using:
  - delayed supervised signal,
  - reservoir replay memory,
  - anchor regularization to reduce forgetting.
- Logs rolling accuracy so you can observe improvement over time.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python train_online.py \
  --num-steps 20000 \
  --classes-per-task 2 \
  --delay-steps 25 \
  --replay-buffer-size 5000 \
  --replay-batch-size 32 \
  --log-every 100 \
  --save-every 2000 \
  --device cuda
```

## Recommended quick smoke test

```bash
python train_online.py --num-steps 300 --log-every 50 --save-every 300
```

## Outputs

- Checkpoints in `./checkpoints/` (`step_*.pt` and `final.pt`).
- Progress bar with rolling accuracy and replay size.
- Structured logs with step metrics.
- Task id in logs/postfix for Split-CIFAR10 phase tracking.

## Notes

- If CUDA is unavailable, the script automatically falls back to CPU.
- This is intentionally a toy implementation meant for experimentation and extension.
