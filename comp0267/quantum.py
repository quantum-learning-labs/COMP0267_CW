"""Quantum text model (a variational circuit per sentence) and an angle-encoding image map.

Gate tensors are indexed ``(inputs..., outputs...)``, so a k-qubit gate ``U`` is stored as ``U.T`` reshaped to
``(2,) * 2k``. A wire label shared by two qubits that are never measured is a cup: the einsum sums over it, which is
the (unnormalised) Bell effect ``<00| + <11|`` on those two qubits.
"""
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import count

import opt_einsum as oe, torch
from torch import nn

from comp0267.diagrams import Diagram, add_root, symbol
from comp0267.einsum import EinsumModel, Op, Recipe, batched_expression

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
        t = t if kind == "0" else gate_tensor(t)
        return t.expand(batch, *t.shape)
    return gate_tensor(GATES[kind](angles))


class CircuitBuilder:
    """Writes a circuit as a recipe. Each qubit carries an einsum label that changes after every gate."""

    def __init__(self) -> None:
        self.subscripts: list[str] = []
        self.ops: list[Op] = []
        self._labels = map(oe.get_symbol, count())

    def qubit(self) -> str:
        """A new qubit in state |0>; returns its label."""
        label = next(self._labels)
        self.subscripts.append(label)
        self.ops.append(Op(None, "0", (2,)))
        return label

    def apply(self, kind: str, wires: list[str], qubits: Sequence[int], name: str | None = None) -> None:
        """Apply gate ``kind`` to ``wires[q] for q in qubits`` (``wires`` is updated); ``name`` labels its angle."""
        if (name is None) != (kind in FIXED):
            raise ValueError(f"gate {kind!r}: fixed gates take no name, parametrised gates need one")
        outs = [next(self._labels) for _ in qubits]
        self.subscripts.append("".join(wires[q] for q in qubits) + "".join(outs))
        self.ops.append(Op(name, kind, (2,) * (2 * len(qubits))))
        for q, out in zip(qubits, outs):
            wires[q] = out

    def recipe(self, outputs: Sequence[str], rename: Mapping[str, str] | None = None) -> Recipe:
        """Finish the circuit: ``outputs`` are the open wires, ``rename`` joins wires into cups."""
        rename = rename or {}
        subscripts = ("".join(rename.get(c, c) for c in s) for s in self.subscripts)
        return Recipe(",".join(subscripts) + "->" + "".join(outputs), tuple(self.ops))


class Ansatz(ABC):
    """Diagram -> Recipe: each box becomes a circuit on ``qubits[type]`` qubits per wire; shared wires become cups.

    With ``root=False`` (default) the output is the head's open sentence wire; with ``root=True`` a shared ``ROOT``
    circuit maps it to ``qubits['out']`` output qubits, as in the classical model.
    """

    def __init__(self, qubits: Mapping[str, int], layers: int = 1, root: bool = False) -> None:
        self.qubits = dict(qubits)
        self.layers = layers
        self.root = root

    @abstractmethod
    def word_circuit(self, circuit: CircuitBuilder, wires: list[str], name: str) -> None:
        """Apply one word's gates to its qubits ``wires``. Name every angle ``f"{name}_..."``."""

    def __call__(self, diagram: Diagram) -> Recipe:
        circuit = CircuitBuilder()
        first_end: dict[int, list[str]] = {}  # diagram wire -> qubits of the first box that has it
        cups: dict[str, str] = {}
        for word, ccg_wires, types in add_root(diagram) if self.root else diagram:
            wires = [circuit.qubit() for t in types for _ in range(self.qubits[t])]
            self.word_circuit(circuit, wires, symbol(word, types))
            start = 0
            for ccg_wire, t in zip(ccg_wires, types):
                part, start = wires[start:start + self.qubits[t]], start + self.qubits[t]
                if ccg_wire in first_end:
                    cups.update(zip(part, first_end.pop(ccg_wire)))
                else:
                    first_end[ccg_wire] = part
        if list(first_end) != [0]:
            raise ValueError(f"diagram wires {sorted(first_end)} are open; only the output wire 0 may be")
        return circuit.recipe(outputs=first_end[0], rename=cups)


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


class AngleEncoder(nn.Module):
    """Image features -> state ``prod_layers [CX ladder] [Rz(x) Ry(x) on every qubit] |0...0>``."""

    def __init__(self, n_qubits: int, layers: int = 1) -> None:
        super().__init__()
        circuit, features = CircuitBuilder(), count()
        wires = [circuit.qubit() for _ in range(n_qubits)]
        for _ in range(layers):
            for q in range(n_qubits):
                circuit.apply("Ry", wires, [q], f"x{next(features)}")
                circuit.apply("Rz", wires, [q], f"x{next(features)}")
            for q in range(n_qubits - 1):
                circuit.apply("CX", wires, [q, q + 1])
        self.recipe = circuit.recipe(outputs=wires)
        self.n_qubits, self.n_features = n_qubits, next(features)
        self.register_buffer("_anchor", torch.zeros(0), persistent=False)  # follows .to(device)

    def angle_dict(self, x: torch.Tensor) -> dict[str, float]:
        """Angles of one image's circuit (to bind into a Qiskit circuit)."""
        return {f"x{j}": float(v) for j, v in enumerate(x)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self._anchor.device)
        operands = [operand(op.kind, len(x), None if op.name is None else x[:, int(op.name[1:])], x.device)
                    for op in self.recipe.ops]
        shapes = tuple(op.shape for op in self.recipe.ops)
        return batched_expression(self.recipe.einsum, shapes)(*operands).flatten(1)
