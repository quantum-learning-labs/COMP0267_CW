"""Similarities, the InfoNCE loss and the two-way choice evaluation."""
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

Similarity = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix of cosine similarities between the rows of ``a`` and ``b``."""
    return F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T


def fidelity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix of fidelities |<a_i|b_j>|^2 between the normalised rows (states) of ``a`` and ``b``."""
    return (F.normalize(a, dim=-1).conj() @ F.normalize(b, dim=-1).T).abs().square()


class InfoNCE(nn.Module):
    """Symmetric InfoNCE (CLIP) loss: in a batch of matched pairs, each caption must pick out its image and vice versa."""

    def __init__(self, similarity: Similarity, temperature: float = 0.07) -> None:
        super().__init__()
        self.similarity, self.temperature = similarity, temperature

    def forward(self, text: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        logits = self.similarity(text, image) / self.temperature
        labels = torch.arange(len(logits), device=logits.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


@torch.no_grad()
def choice_scores(text_model: nn.Module, image_model: nn.Module, loader: DataLoader,
                  similarity: Similarity) -> torch.Tensor:
    """Per-row score of a two-way choice: 1 if the positive beats the negative, 0.5 on a tie, else 0.

    Rows hold a caption, its image and either a negative image or a negative caption.
    """
    text_model.eval()
    image_model.eval()
    scores = []
    for batch in loader:
        text, image = text_model(batch["caption"]), image_model(batch["image"])
        positive = similarity(text, image).diagonal()
        if "negative_image" in batch:
            negative = similarity(text, image_model(batch["negative_image"])).diagonal()
        else:
            negative = similarity(text_model(batch["negative_caption"]), image).diagonal()
        scores.append((positive > negative).float() + 0.5 * (positive == negative).float())
    return torch.cat(scores).cpu()
