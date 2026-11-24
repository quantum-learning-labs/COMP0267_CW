"""Classical tensor-network text model and a PCA image projection."""
import math
from collections.abc import Callable, Iterable, Mapping
from itertools import count

import opt_einsum as oe
import torch
from torch import nn

from modules.diagrams import Diagram, add_root, symbol
from modules.einsum import EinsumModel, Op, Recipe


def mps_cores(name: str, wires: list[int], shape: tuple[int, ...], bond_dim: int,
              new_wire: Callable[[], int]) -> list[tuple[str, list[int], tuple[int, ...]]]:
    """Split an order-k tensor into a chain of k cores joined by k-1 new bond wires of size ``bond_dim``."""
    bonds = [new_wire() for _ in range(len(wires) - 1)]
    left = [None] + bonds
    right = bonds + [None]
    cores = []
    for k, (wire, dim) in enumerate(zip(wires, shape)):
        core_wires = [w for w in (left[k], wire, right[k]) if w is not None]
        core_shape = tuple(bond_dim if w in bonds else dim for w in core_wires)
        cores.append((f"{name}_core{k}", core_wires, core_shape))
    return cores


class TNCompiler:
    """Diagram -> Recipe: each box becomes a tensor with one axis of size ``dims[type]`` per wire.

    With ``bond_dim`` set, boxes with more than two wires are replaced by matrix product states (MPS).
    """

    def __init__(self, dims: Mapping[str, int], bond_dim: int | None = None) -> None:
        self.dims = dict(dims)
        self.bond_dim = bond_dim

    def __call__(self, diagram: Diagram) -> Recipe:
        diagram = add_root(diagram)
        new_wire = count(max(w for _, wires, _ in diagram for w in wires) + 1).__next__
        tensors = []
        for word, wires, types in diagram:
            name, shape = symbol(word, types), tuple(self.dims[t] for t in types)
            if self.bond_dim and len(wires) > 2:
                tensors += mps_cores(name, wires, shape, self.bond_dim, new_wire)
            else:
                tensors.append((name, wires, shape))
        einsum = ",".join("".join(map(oe.get_symbol, wires)) for _, wires, _ in tensors) + "->" + oe.get_symbol(0)
        return Recipe(einsum, tuple(Op(name, "tensor", shape) for name, _, shape in tensors))


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


class PCA(nn.Module):
    """Frozen linear projection onto the top principal components (optionally whitened to unit variance)."""

    def __init__(self, dim: int, whiten: bool = False) -> None:
        super().__init__()
        self.dim, self.whiten = dim, whiten
        self.register_buffer("mean", torch.zeros(0))
        self.register_buffer("components", torch.zeros(0))

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> "PCA":
        self.mean = x.mean(0)
        _, s, vh = torch.linalg.svd(x - self.mean, full_matrices=False)
        self.components = vh[:self.dim].T
        if self.whiten:
            self.components = self.components / (s[:self.dim] / math.sqrt(len(x) - 1))
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.to(self.mean.device) - self.mean) @ self.components
