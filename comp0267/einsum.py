"""Recipes (einsum string + tensor list) and a model that contracts batches of them.

Both text models compile a sentence into a ``Recipe``: an einsum string such as ``'ab,bc,c->a'`` and one ``Op`` per
operand. An ``Op`` names the parameters it is built from, so the same word reuses the same parameters in every
sentence. Sentences with the same structure are contracted together as one batched einsum.
"""
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from functools import cache
from typing import NamedTuple

import opt_einsum as oe
from opt_einsum.contract import ContractExpression
import torch
from torch import nn


class Op(NamedTuple):
    name: str | None        # parameter symbol shared across sentences; None for a fixed tensor
    kind: str               # 'tensor' (classical) or a gate: '0' (|0> state), 'H', 'CX', 'Ry', ...
    shape: tuple[int, ...]


class Recipe(NamedTuple):
    einsum: str
    ops: tuple[Op, ...]


@cache
def batched_expression(einsum: str, shapes: tuple[tuple[int, ...], ...],
                       nominal_batch: int = 64) -> ContractExpression:
    """The einsum with a leading batch axis on every operand and the output, with its contraction path precomputed.

    The path is searched once for ``nominal_batch`` and is valid for any batch size.
    """
    b = next(oe.parser.gen_unused_symbols(einsum, 1))
    inputs, output = einsum.split("->")
    batched = ",".join(b + s for s in inputs.split(",")) + f"->{b}{output}"
    return oe.contract_expression(batched, *[(nominal_batch, *s) for s in shapes])


class EinsumModel(nn.Module, ABC):
    """Contracts a batch of recipes; subclasses decide how each operand is built from parameters."""

    @abstractmethod
    def operand(self, ops: Sequence[Op]) -> torch.Tensor:
        """Stack of the tensors of ``ops`` (one operand slot across a batch): shape ``(len(ops), *shape)``."""

    @abstractmethod
    def symbols(self) -> set[str]:
        """Parameter symbols the model has learned."""

    def knows(self, recipe: Recipe) -> bool:
        """Whether every parametrised op of the recipe has learned parameters."""
        known = self.symbols()
        return all(op.name is None or op.name in known for op in recipe.ops)

    def forward(self, recipes: Sequence[Recipe]) -> torch.Tensor:
        """Contract each recipe; returns ``(len(recipes), output size)`` with the output wires flattened."""
        groups: dict[tuple, list[int]] = defaultdict(list)
        for i, recipe in enumerate(recipes):
            groups[(recipe.einsum, tuple((op.kind, op.shape) for op in recipe.ops))].append(i)

        order, outputs = [], []
        for (einsum, signature), rows in groups.items():
            slots = zip(*(recipes[i].ops for i in rows))
            operands = [self.operand(slot) for slot in slots]
            outputs.append(batched_expression(einsum, tuple(shape for _, shape in signature))(*operands).flatten(1))
            order += rows
        out = torch.cat(outputs)
        return out[torch.argsort(torch.tensor(order, device=out.device))]
