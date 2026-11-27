"""Diagram -> tensor network recipe: a tensor per word (optionally an MPS), shared wires contracted."""
from collections.abc import Callable, Mapping
from itertools import count

import opt_einsum as oe

from comp0267.compilation.einsum import Op, Recipe
from comp0267.symbolic.diagrams import Diagram, add_root, symbol


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
