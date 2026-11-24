"""Recipes -> Qiskit circuits, and the text-image overlap |<image|text>|^2 estimated from them.

A text circuit prepares its sentence state only after post-selection: every cup (pair of qubits joined by a
diagram wire) is measured in the Bell basis and the shot is kept only if both qubits read 0. The overlap with an
image state ``V|0>`` is measured by compute-uncompute: run the text circuit, apply ``V^dagger`` to its output qubits
and count how often the outputs read all 0 among the kept shots.
"""
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import count

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from modules.einsum import Recipe

Counts = dict[str, int]


def to_circuit(recipe: Recipe, angles: Mapping[str, float]) -> QuantumCircuit:
    """Circuit of a recipe with ``angles`` bound. ``metadata`` lists the ``postselect`` and ``outputs`` qubits."""
    inputs, outputs = recipe.einsum.split("->")
    qc = QuantumCircuit(sum(op.kind == "0" for op in recipe.ops))
    qubits_of: dict[str, list[int]] = defaultdict(list)  # wire label -> qubits whose current wire it is
    new_qubit = count()
    for subscript, op in zip(inputs.split(","), recipe.ops):
        if op.kind == "0":
            qubits_of[subscript].append(next(new_qubit))
            continue
        k = len(subscript) // 2
        qubits = [qubits_of.pop(w)[0] for w in subscript[:k]]
        gate = getattr(qc, op.kind.lower())
        gate(*([] if op.name is None else [angles[op.name]]), *qubits)
        for w, q in zip(subscript[k:], qubits):
            qubits_of[w].append(q)

    postselect = []
    for pair in (qs for qs in qubits_of.values() if len(qs) == 2):  # a cup: rotate the Bell state |00>+|11> to |00>
        qc.cx(*pair)
        qc.h(pair[0])
        postselect += pair
    qc.metadata = {"postselect": postselect, "outputs": [qubits_of[w][0] for w in outputs]}
    return qc


def overlap_circuit(text: QuantumCircuit, image: QuantumCircuit, measure: bool = True) -> QuantumCircuit:
    """Text circuit followed by the inverse image circuit on the text's output qubits."""
    qc = text.compose(image.inverse(), qubits=text.metadata["outputs"])
    qc.metadata = dict(text.metadata)
    if measure:
        qc.measure_all()
    return qc


def exact_overlap(qc: QuantumCircuit) -> tuple[float, float]:
    """(post-selection probability, overlap) of an unmeasured overlap circuit, from its statevector."""
    state = Statevector(qc)
    post, out = qc.metadata["postselect"], qc.metadata["outputs"]
    p_accept = float(state.probabilities(post)[0]) if post else 1.0
    return p_accept, float(state.probabilities(post + out)[0]) / p_accept


def estimate_overlap(counts: Counts, qc: QuantumCircuit) -> tuple[float, int]:
    """(overlap estimate, number of accepted shots) from the counts of a measured overlap circuit."""
    post, out = qc.metadata["postselect"], qc.metadata["outputs"]
    accepted = hits = 0
    for bits, n in counts.items():
        bits = bits.replace(" ", "")
        zero = lambda qubits: all(bits[-1 - q] == "0" for q in qubits)  # Qiskit bit strings are little-endian
        if zero(post):
            accepted += n
            hits += n * zero(out)
    return (hits / accepted if accepted else float("nan")), accepted


def sample(circuits: Sequence[QuantumCircuit], backend, shots: int, seed: int | None = None) -> list[Counts]:
    """Transpile for ``backend`` and run; one counts dictionary per circuit."""
    compiled = transpile(list(circuits), backend, optimization_level=1, seed_transpiler=seed)
    result = backend.run(compiled, shots=shots, seed_simulator=seed).result()
    return [result.get_counts(i) for i in range(len(compiled))]
