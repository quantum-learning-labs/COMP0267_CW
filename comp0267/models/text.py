"""Text models: contract a batch of recipes with learned parameters (a tensor network or a variational circuit)."""
import math
from collections.abc import Iterable

import torch
from torch import nn

from comp0267.compilation.einsum import EinsumModel, Recipe
from comp0267.compilation.gates import operand


class TNModel(EinsumModel):
    """One trainable tensor per symbol seen in the training recipes."""

    def __init__(self, recipes: Iterable[Recipe]) -> None:
        super().__init__()
        shapes = {op.name: op.shape for recipe in recipes for op in recipe.ops}
        self.tensors = nn.ParameterDict({name: nn.Parameter(self.init_tensor(shape)) for name, shape in shapes.items()})

    @staticmethod
    def init_tensor(shape: tuple[int, ...]) -> torch.Tensor:
        """Glorot-normal initialisation, treating the first axis as the output."""
        fan_in = math.prod(shape[1:]) if len(shape) > 1 else shape[0]
        return torch.randn(shape) * math.sqrt(2.0 / (shape[0] + fan_in))

    def symbols(self) -> set[str]:
        return set(self.tensors.keys())

    def operand(self, ops):
        return torch.stack([self.tensors[op.name] for op in ops])


class VQCModel(EinsumModel):
    """One trainable angle per symbol seen in the training recipes."""

    def __init__(self, recipes: Iterable[Recipe], init_scale: float = math.pi) -> None:
        super().__init__()
        names = sorted({op.name for recipe in recipes for op in recipe.ops if op.name is not None})
        self.index = {name: i for i, name in enumerate(names)}
        self.angles = nn.Parameter(torch.empty(len(names)).uniform_(-init_scale, init_scale))

    def symbols(self) -> set[str]:
        return set(self.index)

    def angle_dict(self) -> dict[str, float]:
        """Current angles by symbol (to bind into Qiskit circuits)."""
        return dict(zip(self.index, self.angles.detach().cpu().tolist()))

    def operand(self, ops):
        kind, device = ops[0].kind, self.angles.device
        if ops[0].name is None:
            return operand(kind, len(ops), None, device)
        idx = torch.tensor([self.index[op.name] for op in ops], device=device)
        return operand(kind, len(ops), self.angles[idx], device)
