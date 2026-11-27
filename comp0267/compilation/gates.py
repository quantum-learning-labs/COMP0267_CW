"""Gates as torch tensors.

Gate tensors are indexed ``(inputs..., outputs...)``, so a k-qubit gate ``U`` is stored as ``U.T`` reshaped to
``(2,) * 2k``. A qubit post-selected on 0 ends in the effect ``<0|`` (kind ``"post0"``).
"""
import math
from collections.abc import Callable

import torch

DTYPE = torch.complex64


def _matrix(rows: list[list[torch.Tensor]]) -> torch.Tensor:
    return torch.stack([torch.stack(row, -1) for row in rows], -2)


def rx(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta.to(DTYPE) / 2), torch.sin(theta.to(DTYPE) / 2)
    return _matrix([[c, -1j * s], [-1j * s, c]])


def ry(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta.to(DTYPE) / 2), torch.sin(theta.to(DTYPE) / 2)
    return _matrix([[c, -s], [s, c]])


def rz(theta: torch.Tensor) -> torch.Tensor:
    e = torch.exp(-0.5j * theta.to(DTYPE))
    return _matrix([[e, torch.zeros_like(e)], [torch.zeros_like(e), 1 / e]])


def controlled(u: torch.Tensor) -> torch.Tensor:
    """``|0><0| (x) I + |1><1| (x) U``, control on the first qubit."""
    out = torch.zeros(*u.shape[:-2], 4, 4, dtype=u.dtype, device=u.device)
    out[..., 0, 0] = out[..., 1, 1] = 1
    out[..., 2:, 2:] = u
    return out


# parametrised gates: batch of angles -> batch of unitaries; names match QuantumCircuit methods (lower-cased)
GATES: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "Rx": rx, "Ry": ry, "Rz": rz,
    "CRx": lambda t: controlled(rx(t)), "CRy": lambda t: controlled(ry(t)), "CRz": lambda t: controlled(rz(t)),
}
FIXED: dict[str, torch.Tensor] = {
    "0": torch.tensor([1, 0], dtype=DTYPE),
    "post0": torch.tensor([1, 0], dtype=DTYPE),  # the effect <0|: post-select a qubit on 0
    "H": torch.tensor([[1, 1], [1, -1]], dtype=DTYPE) / math.sqrt(2),
    "CX": torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=DTYPE),
    "CZ": torch.diag(torch.tensor([1, 1, 1, -1], dtype=DTYPE)),
}


def gate_tensor(u: torch.Tensor) -> torch.Tensor:
    """Unitary ``(..., 2^k, 2^k)`` -> einsum tensor ``(..., 2, ..., 2)`` indexed (inputs, outputs)."""
    k = u.shape[-1].bit_length() - 1
    return u.transpose(-1, -2).reshape(*u.shape[:-2], *(2,) * (2 * k))


def operand(kind: str, batch: int, angles: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    """A batch of one gate's tensors: fixed gates are repeated, parametrised gates are built from ``angles``."""
    if angles is None:
        t = FIXED[kind].to(device)
        t = t if kind in ("0", "post0") else gate_tensor(t)
        return t.expand(batch, *t.shape)
    return gate_tensor(GATES[kind](angles))
