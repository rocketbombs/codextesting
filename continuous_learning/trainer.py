import logging
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from continuous_learning.config import OnlineConfig
from continuous_learning.memory import ReservoirReplayBuffer
from continuous_learning.model import TinyCNN
from continuous_learning.stream import DelayedLabelQueue, OnlineSplitCIFAR10Stream


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


def smoothed_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_smoothing: float,
    fallback_warned: bool,
) -> tuple[torch.Tensor, bool]:
    if label_smoothing <= 0.0:
        return F.cross_entropy(logits, targets), fallback_warned

    try:
        return (
            F.cross_entropy(logits, targets, label_smoothing=label_smoothing),
            fallback_warned,
        )
    except TypeError:
        if not fallback_warned:
            logging.warning(
                "torch.nn.functional.cross_entropy does not support label_smoothing "
                "in this runtime; using manual smoothing fallback."
            )
            fallback_warned = True

        num_classes = logits.size(1)
        if num_classes <= 1:
            return F.cross_entropy(logits, targets), fallback_warned

        smooth = label_smoothing / (num_classes - 1)
        true_dist = torch.full_like(logits, smooth)
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing)
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(true_dist * log_probs).sum(dim=1).mean()
        return loss, fallback_warned


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
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.num_steps),
        eta_min=config.lr * config.min_lr_ratio,
    )
    stream = OnlineSplitCIFAR10Stream(
        data_dir=config.data_dir,
        batch_size=64,
        num_workers=config.num_workers,
        total_steps=config.num_steps,
        classes_per_task=config.classes_per_task,
    )
    label_queue = DelayedLabelQueue(delay_steps=config.delay_steps)
    replay = ReservoirReplayBuffer(capacity=config.replay_buffer_size, seed=config.seed)

    rolling_correct = deque(maxlen=config.eval_window)
    last_ce = float("nan")
    last_main_ce = float("nan")
    last_replay_ce = float("nan")
    last_entropy = float("nan")
    last_entropy_reg = float("nan")
    smoothing_fallback_warned = False

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
                rx = rx.unsqueeze(0).to(device)
                ry_t = torch.tensor([ry], dtype=torch.long, device=device)

                model.train()
                optimizer.zero_grad(set_to_none=True)

                main_logits = model(rx)
                main_ce, smoothing_fallback_warned = smoothed_cross_entropy(
                    main_logits,
                    ry_t,
                    label_smoothing=config.label_smoothing,
                    fallback_warned=smoothing_fallback_warned,
                )

                replay_ce_sum = torch.tensor(0.0, device=device)
                replay_ce_count = 0
                for _ in range(max(0, config.replay_updates_per_step)):
                    replay_batch = replay.sample(config.replay_batch_size, device=device)
                    if replay_batch is None:
                        continue
                    replay_x, replay_y = replay_batch
                    replay_logits = model(replay_x)
                    replay_ce_i, smoothing_fallback_warned = smoothed_cross_entropy(
                        replay_logits,
                        replay_y,
                        label_smoothing=config.label_smoothing,
                        fallback_warned=smoothing_fallback_warned,
                    )
                    replay_ce_sum = replay_ce_sum + replay_ce_i
                    replay_ce_count += 1

                replay_ce = replay_ce_sum / max(1, replay_ce_count)
                ce_loss = main_ce + config.replay_loss_weight * replay_ce
                probs = F.softmax(main_logits, dim=1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                entropy_reg = -config.entropy_lambda * entropy

                l2_anchor = 0.0
                for p, p0 in zip(model.parameters(), anchor_params):
                    l2_anchor = l2_anchor + torch.sum((p - p0.to(device)) ** 2)

                loss = ce_loss + entropy_reg + config.l2_anchor_lambda * l2_anchor
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                replay.add(rx.squeeze(0), int(ry))
                last_ce = float(ce_loss.item())
                last_main_ce = float(main_ce.item())
                last_replay_ce = float(replay_ce.item())
                last_entropy = float(entropy.item())
                last_entropy_reg = float(entropy_reg.item())

        if step % config.log_every == 0:
            rolling_acc = np.mean(rolling_correct) * 100 if rolling_correct else 0.0
            pbar.set_postfix(
                rolling_acc=f"{rolling_acc:.2f}%",
                replay_size=len(replay),
                delay=config.delay_steps,
                task=sample.task_id,
            )
            logging.info(
                "step=%d task=%d rolling_acc=%.2f%% replay_size=%d lr=%.6e ce=%.4f main_ce=%.4f replay_ce=%.4f entropy=%.4f entropy_reg=%.6f",
                step,
                sample.task_id,
                rolling_acc,
                len(replay),
                optimizer.param_groups[0]["lr"],
                last_ce,
                last_main_ce,
                last_replay_ce,
                last_entropy,
                last_entropy_reg,
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
        replay.add(rx, int(ry))

    final_ckpt = os.path.join(config.ckpt_dir, "final.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": config.__dict__}, final_ckpt)
    logging.info("training complete. final model saved to %s", final_ckpt)
