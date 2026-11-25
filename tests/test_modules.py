"""Consistency checks: torch contractions agree with Qiskit statevectors. Run with ``python -m pytest tests``."""
import math

import numpy as np
import pytest
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from comp0267.circuits import estimate_overlap, exact_overlap, overlap_circuit, to_circuit
from comp0267.classical import TNCompiler, TNModel
from comp0267.diagrams import add_root
from comp0267.quantum import GATES, AngleEncoder, Ansatz, VQCModel
from comp0267.training import fidelity

# "alice loves bob": the verb has wires (n, s, n); wire 0 is the sentence output
DIAGRAM = [("alice", [1], ["n"]), ("loves", [1, 0, 2], ["n", "s", "n"]), ("bob", [2], ["n"])]


class RyCX(Ansatz):
    def word_circuit(self, circuit, wires, name):
        for layer in range(self.layers):
            for q in range(len(wires)):
                circuit.apply("Ry", wires, [q], f"{name}_l{layer}_q{q}")
                circuit.apply("Rz", wires, [q], f"{name}_l{layer}_z{q}")
            for q in range(len(wires) - 1):
                circuit.apply("CX", wires, [q, q + 1])


@pytest.mark.parametrize("kind", sorted(GATES))
def test_gates_match_qiskit(kind):
    theta = 0.7
    qc = QuantumCircuit(2 if kind.startswith("C") else 1)
    getattr(qc, kind.lower())(theta, *range(qc.num_qubits))
    expected = Operator(qc).reverse_qargs().data  # big-endian: first qubit most significant, as in our tensors
    assert np.allclose(GATES[kind](torch.tensor(theta)).numpy(), expected, atol=1e-6)


def test_add_root_closes_every_wire_but_the_output():
    rooted = add_root(DIAGRAM)
    wires = [w for _, ws, _ in rooted for w in ws]
    assert rooted[-1][0] == "ROOT" and rooted[-1][2] == ["s", "out"]
    assert [w for w in set(wires) if wires.count(w) == 1] == [0]


def test_tn_model_output_shape():
    recipe = TNCompiler({"n": 3, "s": 2, "out": 5}, bond_dim=2)(DIAGRAM)
    assert TNModel([recipe])([recipe, recipe]).shape == (2, 5)


@pytest.mark.parametrize("qubits, root", [({"n": 1, "s": 1, "out": 1}, True), ({"n": 1, "s": 2, "out": 2}, True),
                                          ({"n": 1, "s": 2}, False)])
def test_vqc_matches_postselected_statevector(qubits, root):
    torch.manual_seed(0)
    recipe = RyCX(qubits, layers=2, root=root)(DIAGRAM)
    model = VQCModel([recipe])
    state = model([recipe])[0].detach().numpy()

    qc = to_circuit(recipe, model.angle_dict())
    post, out = qc.metadata["postselect"], qc.metadata["outputs"]
    amplitudes = Statevector(qc).data.reshape([2] * qc.num_qubits)  # axis k is qubit n-1-k
    index = [0 if q in post else slice(None) for q in reversed(range(qc.num_qubits))]
    kept = [q for q in reversed(range(qc.num_qubits)) if q not in post]
    projected = np.transpose(amplitudes[tuple(index)], [kept.index(q) for q in out]).reshape(-1)
    # each cup is <00|+<11| in the einsum but (<00|+<11|)/sqrt(2) in the circuit
    assert np.allclose(state, projected * 2 ** (len(post) / 4), atol=1e-5)


def test_angle_encoder_matches_qiskit():
    encoder = AngleEncoder(n_qubits=2, layers=2)
    x = torch.randn(3, encoder.n_features)
    states = encoder(x).numpy()
    for x_i, state in zip(x, states):
        qc = to_circuit(encoder.recipe, encoder.angle_dict(x_i))
        sv = Statevector(qc).data.reshape(2, 2).T.reshape(-1)  # to big-endian
        assert np.allclose(state, sv, atol=1e-5)


def test_exact_overlap_matches_torch_fidelity():
    torch.manual_seed(1)
    recipe = RyCX({"n": 1, "s": 2}, layers=1)(DIAGRAM)
    model, encoder = VQCModel([recipe]), AngleEncoder(n_qubits=2, layers=1)
    x = torch.randn(1, encoder.n_features)
    expected = fidelity(model([recipe]), encoder(x)).item()

    qc = overlap_circuit(to_circuit(recipe, model.angle_dict()), to_circuit(encoder.recipe, encoder.angle_dict(x[0])),
                         measure=False)
    p_accept, overlap = exact_overlap(qc)
    assert math.isclose(overlap, expected, abs_tol=1e-5)
    norm2 = model([recipe]).abs().square().sum().item()
    assert math.isclose(p_accept, norm2 / 2 ** (len(qc.metadata["postselect"]) // 2), rel_tol=1e-4)


def test_ansatz_rejects_dangling_wires():
    with pytest.raises(ValueError):
        RyCX({"n": 1, "s": 1})(DIAGRAM[:2])  # "alice loves": the object wire is left open


def test_estimate_overlap_reads_little_endian_counts():
    qc = QuantumCircuit(3)
    qc.metadata = {"postselect": [0, 1], "outputs": [2]}
    counts = {"000": 30, "100": 10, "001": 50}  # bit q is the q-th character from the right
    assert estimate_overlap(counts, qc) == (0.75, 40)
