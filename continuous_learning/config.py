from dataclasses import dataclass


@dataclass
class OnlineConfig:
    data_dir: str = "./data"
    num_steps: int = 20_000
    eval_window: int = 500
    replay_buffer_size: int = 5_000
    replay_batch_size: int = 32
    update_every: int = 1
    delay_steps: int = 25
    lr: float = 1e-3
    min_lr_ratio: float = 0.05
    weight_decay: float = 1e-4
    l2_anchor_lambda: float = 1e-4
    label_smoothing: float = 0.05
    entropy_lambda: float = 1e-4
    replay_loss_weight: float = 1.0
    replay_updates_per_step: int = 2
    num_workers: int = 2
    device: str = "cuda"
    seed: int = 42
    log_every: int = 100
    save_every: int = 2_000
    ckpt_dir: str = "./checkpoints"
