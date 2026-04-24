import random
from dataclasses import dataclass

import torch


@dataclass
class MemoryItem:
    image: torch.Tensor
    label: int


class ReservoirReplayBuffer:
    def __init__(self, capacity: int, seed: int = 42) -> None:
        self.capacity = capacity
        self.items: list[MemoryItem] = []
        self.seen = 0
        self.rng = random.Random(seed)

    def add(self, image: torch.Tensor, label: int) -> None:
        self.seen += 1
        image = image.detach().cpu().clone()
        if len(self.items) < self.capacity:
            self.items.append(MemoryItem(image=image, label=label))
            return

        idx = self.rng.randint(0, self.seen - 1)
        if idx < self.capacity:
            self.items[idx] = MemoryItem(image=image, label=label)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.items:
            return None

        k = min(batch_size, len(self.items))
        batch = self.rng.sample(self.items, k=k)
        images = torch.stack([item.image for item in batch], dim=0).to(device)
        labels = torch.tensor([item.label for item in batch], dtype=torch.long, device=device)
        return images, labels

    def __len__(self) -> int:
        return len(self.items)
