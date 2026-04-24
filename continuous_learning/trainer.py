import logging
import os
import random
from collections import deque

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from continuous_learning.config import OnlineConfig
from continuous_learning.memory import ReservoirReplayBuffer
from continuous_learning.model import TinyCNN
from continuous_learning.stream import DelayedLabelQueue, OnlineMNISTStream


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def online_train(config: OnlineConfig) -> None:
    setup_logger()
    set_seed(config.seed)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logging.warning("CUDA not available; falling back to CPU.")

    os.makedirs(config.ckpt_dir, exist_ok=True)

    model = TinyCNN().to(device)
    anchor_params = [p.detach().clone() for p in model.parameters()]
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    stream = OnlineMNISTStream(
        data_dir=config.data_dir,
        batch_size=64,
        num_workers=config.num_workers,
        total_steps=config.num_steps,
    )
    label_queue = DelayedLabelQueue(delay_steps=config.delay_steps)
    replay = ReservoirReplayBuffer(capacity=config.replay_buffer_size, seed=config.seed)

    rolling_correct = deque(maxlen=config.eval_window)
    num_updates = 0

    def train_on_labeled_sample(rx: torch.Tensor, ry: int) -> None:
        nonlocal num_updates
        rx_b = rx.unsqueeze(0).to(device)
        ry_t = torch.tensor([ry], dtype=torch.long, device=device)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        main_loss = criterion(model(rx_b), ry_t)

        replay_loss = 0.0
        replay_batch = replay.sample(config.replay_batch_size, device=device)
        if replay_batch is not None:
            replay_x, replay_y = replay_batch
            replay_loss = criterion(model(replay_x), replay_y)

        l2_anchor = 0.0
        for p, p0 in zip(model.parameters(), anchor_params):
            l2_anchor = l2_anchor + torch.sum((p - p0.to(device)) ** 2)

        loss = main_loss + 0.7 * replay_loss + config.l2_anchor_lambda * l2_anchor
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        replay.add(rx, int(ry))
        num_updates += 1

    pbar = tqdm(stream.iter_samples(config.num_steps), total=config.num_steps, desc="Online train")
    for step, sample in enumerate(pbar, start=1):
        x = sample.x.unsqueeze(0).to(device)
        y = torch.tensor([sample.y], dtype=torch.long, device=device)

        model.eval()
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct = int((pred == y).item())
            rolling_correct.append(correct)

        label_queue.push(sample.x, sample.y)

        if step % config.update_every == 0:
            ready = label_queue.pop_ready()
            if ready is not None:
                rx, ry = ready
                train_on_labeled_sample(rx, ry)

        if step % config.log_every == 0:
            rolling_acc = np.mean(rolling_correct) * 100 if rolling_correct else 0.0
            pbar.set_postfix(
                rolling_acc=f"{rolling_acc:.2f}%",
                replay_size=len(replay),
                delay=config.delay_steps,
                updates=num_updates,
            )
            logging.info(
                "step=%d rolling_acc=%.2f%% replay_size=%d updates=%d",
                step,
                rolling_acc,
                len(replay),
                num_updates,
            )

        if step % config.save_every == 0:
            ckpt_path = os.path.join(config.ckpt_dir, f"step_{step}.pt")
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config.__dict__,
                },
                ckpt_path,
            )
            logging.info("saved checkpoint: %s", ckpt_path)

    for rx, ry in label_queue.flush_all():
        train_on_labeled_sample(rx, int(ry))

    if num_updates == 0:
        logging.warning(
            "No gradient updates were performed. Increase --num-steps above --delay-steps "
            "(current num_steps=%d, delay_steps=%d).",
            config.num_steps,
            config.delay_steps,
        )

    final_ckpt = os.path.join(config.ckpt_dir, "final.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": config.__dict__}, final_ckpt)
    logging.info("training complete. final model saved to %s", final_ckpt)
