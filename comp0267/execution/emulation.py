"""Fake IBM and IQM devices and their emulation with Aer, cut down to the qubits the circuits need."""
import importlib
from collections import defaultdict
from collections.abc import Sequence

import torch
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from tqdm.auto import tqdm

from comp0267.execution.circuits import Counts, measured

IQM_DEVICES = ("adonis", "aphrodite", "apollo", "garnet")


def fake_backend(name: str):
    """A fake device with a real device's gates, connectivity and noise: IBM (e.g. ``'torino'``) or IQM
    (``'adonis'``, ``'aphrodite'``, ``'apollo'``, ``'garnet'``)."""
    name = name.lower()
    if name in IQM_DEVICES:
        module = importlib.import_module(f"iqm.qiskit_iqm.fake_backends.fake_{name}")
        return getattr(module, f"IQMFake{name.capitalize()}")()
    from qiskit_ibm_runtime import fake_provider
    return getattr(fake_provider, f"Fake{name.capitalize()}")()


def device_noise(device, n_qubits: int) -> NoiseModel:
    """The device's noise model (IQM devices carry their own) restricted to its first ``n_qubits`` qubits."""
    full = getattr(device, "noise_model", None) or NoiseModel.from_backend(device)
    noise = NoiseModel(basis_gates=full.basis_gates)
    for gate, errors in full._local_quantum_errors.items():
        for qubits, error in errors.items():
            if max(qubits) < n_qubits:
                noise.add_quantum_error(error, gate, qubits)
    for gate, error in full._default_quantum_errors.items():
        noise.add_all_qubit_quantum_error(error, gate)
    for qubits, error in full._local_readout_errors.items():
        if max(qubits) < n_qubits:
            noise.add_readout_error(error, qubits)
    if full._default_readout_error is not None:
        noise.add_all_qubit_readout_error(full._default_readout_error)
    noise._custom_noise_passes = full._custom_noise_passes  # e.g. relaxation during delays
    return noise


class Emulator:
    """Aer emulation of ``device`` cut down to its first ``n_qubits`` qubits: their couplings, the device's native gates
    and, if ``noisy``, the device's noise on those qubits only. Without a device: an ideal simulator.

    Circuits are transpiled onto those ``n_qubits`` physical qubits, so the emulator never simulates more. With noise,
    an n-qubit circuit is simulated once, exactly, as a density matrix (4^n entries) when that is cheaper than one
    statevector (2^n entries) per shot, i.e. when 2^n <= shots, and the density matrix fits in memory; otherwise as
    noisy statevector trajectories, one per shot.
    """

    def __init__(self, device=None, n_qubits: int | None = None, noisy: bool = False) -> None:
        gpu = torch.cuda.is_available() and "GPU" in AerSimulator().available_devices()  # a GPU build alone is not a GPU
        options = {"method": "statevector", "device": "GPU" if gpu else "CPU"}
        self.coupling_map = self.basis_gates = self.density_matrix = None
        if device is not None:
            n = min(n_qubits or device.num_qubits, device.num_qubits)
            self.coupling_map = CouplingMap([e for e in device.coupling_map.get_edges() if max(e) < n])
            self.basis_gates = list(device.operation_names)
            if noisy:
                noise = device_noise(device, n)
                # wider noisy circuits: CPU trajectories with shot branching (GPU shot branching ran out of memory on Colab)
                options.update(device="CPU", shot_branching_enable=True, noise_model=noise)
                self.density_matrix = AerSimulator(method="density_matrix", device="CPU", noise_model=noise)
        self.simulator = AerSimulator(**options)

    def _simulator(self, qc: QuantumCircuit, shots: int) -> AerSimulator:
        n = len({q for instruction in qc.data for q in instruction.qubits})
        exact = self.density_matrix is not None and 2 ** n <= shots and n <= self.density_matrix.num_qubits
        return self.density_matrix if exact else self.simulator

    def run(self, circuits: Sequence[QuantumCircuit], shots: int, seed: int | None = None,
            batch_size: int = 30) -> list[Counts]:
        """Measure every qubit, transpile onto the emulated qubits and sample; one counts dictionary per circuit."""
        counts: list[Counts] = []
        with tqdm(total=len(circuits), desc="emulating", unit="circuit") as progress:
            for start in range(0, len(circuits), batch_size):
                batch = measured(circuits[start:start + batch_size])
                if self.coupling_map is not None:  # one process: forking a large notebook process can exhaust RAM
                    batch = transpile(batch, coupling_map=self.coupling_map, basis_gates=self.basis_gates,
                                      optimization_level=1, seed_transpiler=seed, num_processes=1)
                groups: dict[int, list[int]] = defaultdict(list)  # simulator -> positions in the batch
                simulators = {}
                for i, qc in enumerate(batch):
                    sim = self._simulator(qc, shots)
                    simulators[id(sim)] = sim
                    groups[id(sim)].append(i)
                out: list[Counts] = [{}] * len(batch)
                for key, idx in groups.items():
                    result = simulators[key].run([batch[i] for i in idx], shots=shots,
                                                 seed_simulator=None if seed is None else seed + start).result()
                    for j, i in enumerate(idx):
                        out[i] = result.get_counts(j)
                counts += out
                progress.update(len(batch))
        return counts
