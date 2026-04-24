from collections import deque
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


class ProgressiveDriftTransform:
    """Applies stronger augmentation earlier and gradually relaxes over time."""

    def __init__(self, total_steps: int) -> None:
        self.total_steps = total_steps

    def __call__(self, image, step: int) -> torch.Tensor:
        progress = min(max(step / max(1, self.total_steps), 0.0), 1.0)

        max_rotation = 35.0 * (1.0 - 0.75 * progress)
        jitter_strength = 0.5 * (1.0 - 0.8 * progress)
        blur_prob = 0.55 * (1.0 - 0.8 * progress)

        aug = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=max_rotation,
                    translate=(0.12, 0.12),
                    scale=(0.85, 1.15),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.ColorJitter(
                    brightness=jitter_strength,
                    contrast=jitter_strength,
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))],
                    p=blur_prob,
                ),
                transforms.ToTensor(),
            ]
        )
        return aug(image)


@dataclass
class StreamSample:
    x: torch.Tensor
    y: int
    task_id: int


class OnlineSplitCIFAR10Stream:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        total_steps: int,
        classes_per_task: int,
    ) -> None:
        self.transform = ProgressiveDriftTransform(total_steps=total_steps)
        self.dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        self.classes_per_task = max(1, classes_per_task)
        class_order = list(range(10))
        self.task_splits = [
            class_order[i : i + self.classes_per_task]
            for i in range(0, len(class_order), self.classes_per_task)
        ]

        targets = torch.tensor(self.dataset.targets, dtype=torch.long)
        self.task_loaders = []
        for split in self.task_splits:
            mask = torch.zeros_like(targets, dtype=torch.bool)
            for cls in split:
                mask = mask | (targets == cls)
            indices = torch.nonzero(mask, as_tuple=False).squeeze(1).tolist()
            subset = torch.utils.data.Subset(self.dataset, indices)
            loader = DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=True,
            )
            self.task_loaders.append(loader)

    def iter_samples(self, total_steps: int):
        if not self.task_loaders:
            return

        step = 0
        num_tasks = len(self.task_loaders)
        base_steps_per_task = total_steps // num_tasks
        extra = total_steps % num_tasks

        for task_id, loader in enumerate(self.task_loaders):
            steps_for_task = base_steps_per_task + (1 if task_id < extra else 0)
            if steps_for_task <= 0:
                continue

            produced = 0
            while produced < steps_for_task:
                for images, labels in loader:
                    for i in range(images.size(0)):
                        if produced >= steps_for_task or step >= total_steps:
                            break
                        pil_img = transforms.ToPILImage()(images[i])
                        x = self.transform(pil_img, step=step)
                        y = int(labels[i].item())
                        yield StreamSample(x=x, y=y, task_id=task_id)
                        step += 1
                        produced += 1
                    if produced >= steps_for_task or step >= total_steps:
                        break


class DelayedLabelQueue:
    def __init__(self, delay_steps: int) -> None:
        self.delay_steps = delay_steps
        self.queue = deque()

    def push(self, x: torch.Tensor, y: int) -> None:
        self.queue.append((x.detach().cpu(), y))

    def pop_ready(self):
        if len(self.queue) > self.delay_steps:
            return self.queue.popleft()
        return None

    def flush_all(self):
        while self.queue:
            yield self.queue.popleft()
