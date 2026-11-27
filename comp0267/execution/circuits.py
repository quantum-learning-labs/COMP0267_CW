"""Recipes -> Qiskit circuits, and post-selection of their measured counts.

A text circuit prepares its sentence state only after post-selection: a shot is kept only if every post-selected
qubit reads 0. The kept shots' output bits are samples of the sentence state in the computational basis.
"""
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import count

from qiskit import QuantumCircuit

from comp0267.compilation.einsum import Recipe

Counts = dict[str, int]


def to_circuit(recipe: Recipe, angles: Mapping[str, float]) -> QuantumCircuit:
    """Circuit of a recipe with ``angles`` bound. ``metadata`` lists the ``postselect`` and ``outputs`` qubits."""
    inputs, outputs = recipe.einsum.split("->")
    qc = QuantumCircuit(sum(op.kind == "0" for op in recipe.ops))
    qubit_of: dict[str, int] = {}  # current label -> qubit
    new_qubit = count()
    postselected: list[int] = []
    for subscript, op in zip(inputs.split(","), recipe.ops):
        if op.kind == "0":
            qubit_of[subscript] = next(new_qubit)
        elif op.kind == "post0":
            postselected.append(qubit_of.pop(subscript))
        else:
            k = len(subscript) // 2
            qubits = [qubit_of.pop(w) for w in subscript[:k]]
            getattr(qc, op.kind.lower())(*([] if op.name is None else [angles[op.name]]), *qubits)
            qubit_of.update(zip(subscript[k:], qubits))
    qc.metadata = {"postselect": postselected, "outputs": [qubit_of[w] for w in outputs]}
    return qc


def postselect(counts: Counts, qc: QuantumCircuit) -> Counts:
    """Counts of the output bits over the shots whose post-selected qubits all read 0 (every qubit measured)."""
    post, out = qc.metadata["postselect"], qc.metadata["outputs"]
    kept: Counts = defaultdict(int)
    for bits, n in counts.items():
        bits = bits.replace(" ", "")[::-1]  # Qiskit bit strings are little-endian: bits[q] is qubit q
        if all(bits[q] == "0" for q in post):
            kept["".join(bits[q] for q in out)] += n
    return dict(kept)


def measured(circuits: Sequence[QuantumCircuit]) -> list[QuantumCircuit]:
    """Copies of the circuits with every qubit measured (``postselect`` reads all bits)."""
    out = []
    for qc in circuits:
        qc = qc.copy()
        qc.measure_all()
        out.append(qc)
    return out
