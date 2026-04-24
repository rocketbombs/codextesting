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


class OnlineMNISTStream:
    def __init__(self, data_dir: str, batch_size: int, num_workers: int, total_steps: int) -> None:
        self.transform = ProgressiveDriftTransform(total_steps=total_steps)
        self.dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )

    def iter_samples(self, total_steps: int):
        step = 0
        while step < total_steps:
            for images, labels in self.loader:
                for i in range(images.size(0)):
                    if step >= total_steps:
                        return
                    pil_img = transforms.ToPILImage()(images[i])
                    x = self.transform(pil_img, step=step)
                    y = int(labels[i].item())
                    yield StreamSample(x=x, y=y)
                    step += 1


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
