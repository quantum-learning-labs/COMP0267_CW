"""Image model: CLIP embeddings -> quantum states."""
from abc import ABC, abstractmethod

import torch
from torch import nn

from comp0267.compilation.ansatz import CircuitBuilder
from comp0267.compilation.einsum import batched_expression
from comp0267.compilation.gates import operand


class FeatureMap(nn.Module, ABC):
    """Image embeddings -> states on ``n_qubits`` qubits. A subclass says how an embedding becomes angles (``map``)
    and which gates use them (``image_circuit``): the angle named ``x{j}`` is column ``j`` of ``map``'s output.
    """

    def __init__(self, n_qubits: int, layers: int = 1) -> None:
        super().__init__()
        self.n_qubits, self.layers = n_qubits, layers
        circuit = CircuitBuilder()
        wires = [circuit.qubit() for _ in range(n_qubits)]
        self.image_circuit(circuit, wires)
        self.recipe = circuit.recipe(outputs=wires)
        names = {op.name for op in self.recipe.ops if op.name is not None}
        self.n_features = len(names)
        if names != {f"x{j}" for j in range(self.n_features)}:
            raise ValueError(f"image_circuit must name its angles x0, x1, ...; got {sorted(names)}")
        self.register_buffer("_anchor", torch.zeros(0), persistent=False)  # follows .to(device)

    @abstractmethod
    def map(self, embeddings: torch.Tensor) -> torch.Tensor:
        """A batch of embeddings ``(batch, dim)`` -> angles ``(batch, n_features)``."""

    @abstractmethod
    def image_circuit(self, circuit: CircuitBuilder, wires: list[str]) -> None:
        """Apply the image's gates to ``wires``; name the angle taken from column ``j`` of ``map`` ``f"x{j}"``."""

    @torch.no_grad()
    def angle_dict(self, embedding: torch.Tensor) -> dict[str, float]:
        """Angles of one image's circuit (to bind into a Qiskit circuit)."""
        x = self.map(embedding.to(self._anchor.device)[None])[0]
        return {f"x{j}": float(v) for j, v in enumerate(x)}

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        x = self.map(embeddings.to(self._anchor.device))
        operands = [operand(op.kind, len(x), None if op.name is None else x[:, int(op.name[1:])], x.device)
                    for op in self.recipe.ops]
        shapes = tuple(op.shape for op in self.recipe.ops)
        return batched_expression(self.recipe.einsum, shapes)(*operands).flatten(1)
