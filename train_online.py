import argparse

from continuous_learning.config import OnlineConfig
from continuous_learning.trainer import online_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online continuous learning on Split-CIFAR10 stream")
    parser.add_argument("--num-steps", type=int, default=20_000)
    parser.add_argument("--classes-per-task", type=int, default=2)
    parser.add_argument("--delay-steps", type=int, default=25)
    parser.add_argument("--replay-buffer-size", type=int, default=5_000)
    parser.add_argument("--replay-batch-size", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=2_000)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--entropy-lambda", type=float, default=1e-4)
    parser.add_argument("--replay-loss-weight", type=float, default=1.0)
    parser.add_argument("--replay-updates-per-step", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OnlineConfig(
        num_steps=args.num_steps,
        classes_per_task=args.classes_per_task,
        delay_steps=args.delay_steps,
        replay_buffer_size=args.replay_buffer_size,
        replay_batch_size=args.replay_batch_size,
        log_every=args.log_every,
        save_every=args.save_every,
        data_dir=args.data_dir,
        ckpt_dir=args.ckpt_dir,
        device=args.device,
        lr=args.lr,
        min_lr_ratio=args.min_lr_ratio,
        label_smoothing=args.label_smoothing,
        entropy_lambda=args.entropy_lambda,
        replay_loss_weight=args.replay_loss_weight,
        replay_updates_per_step=args.replay_updates_per_step,
    )
    online_train(cfg)


if __name__ == "__main__":
    main()
