"""Real quantum hardware: connect to IBM Quantum or IQM Resonance, submit circuits and retrieve their counts later.

Queues can be long, so ``submit`` returns as soon as the jobs are queued and saves their IDs to a small JSON file;
``retrieve`` reads that file (also after a kernel restart) and waits for the results. Counts come back one dictionary
per circuit, in the order the circuits were submitted, ready for ``comp0267.execution.circuits.postselect``.
"""
import json, os, time
from collections.abc import Sequence
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from tqdm.auto import tqdm

from comp0267.execution.circuits import Counts, measured

IQM_URL = "https://resonance.iqm.tech/"
IQM_MAX_CIRCUITS = 100  # IQM accepts at most this many circuits per job


def ibm_backend(device: str | None = None, token: str | None = None, instance: str | None = None):
    """An IBM Quantum device (e.g. ``'ibm_kingston'``; the least busy one if ``device`` is None). Without ``token``,
    the account saved with ``QiskitRuntimeService.save_account`` is used; the token is never written to disk here."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    instance = instance or None
    if token:
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=instance)
    else:
        service = QiskitRuntimeService(instance=instance)
    return service.backend(device) if device else service.least_busy(operational=True, simulator=False)


def iqm_backend(device: str, token: str | None = None, url: str = IQM_URL):
    """An IQM Resonance device (e.g. ``'garnet'``). IQM's client reads the token from ``IQM_TOKEN`` and refuses a token
    given both there and as an argument, so a given ``token`` is put in ``IQM_TOKEN``."""
    from iqm.qiskit_iqm import IQMProvider

    if token:
        os.environ["IQM_TOKEN"] = token
    return IQMProvider(url, quantum_computer=device.lower()).get_backend()


def _is_ibm(backend) -> bool:
    """A real IBM Quantum device (fake IBM devices run locally through ``backend.run``)."""
    from qiskit_ibm_runtime import IBMBackend

    return isinstance(backend, IBMBackend)


def prepare(circuits: Sequence[QuantumCircuit], backend, optimization_level: int = 1,
            seed: int | None = None) -> list[QuantumCircuit]:
    """Measure every qubit and transpile for ``backend``."""
    return transpile(measured(circuits), backend, optimization_level=optimization_level, seed_transpiler=seed,
                     num_processes=1)


def submit(circuits: Sequence[QuantumCircuit], backend, shots: int, path: str | Path | None = None,
           batch_size: int | None = None, optimization_level: int = 1, seed: int | None = None) -> Path:
    """Measure, transpile and queue ``circuits`` on ``backend`` in jobs of ``batch_size`` circuits (default: one job on
    IBM, 100 per job otherwise). Returns the JSON file (default ``jobs/<device>_<time>.json``) with the job IDs."""
    isa = prepare(circuits, backend, optimization_level, seed)
    ibm = _is_ibm(backend)
    batch_size = batch_size or (len(isa) if ibm else IQM_MAX_CIRCUITS)
    if ibm:
        from qiskit_ibm_runtime import SamplerV2
        sampler = SamplerV2(mode=backend)
    job_ids = []
    for start in tqdm(range(0, len(isa), batch_size), desc="submitting", unit="job"):
        batch = isa[start:start + batch_size]
        job = sampler.run(batch, shots=shots) if ibm else backend.run(batch, shots=shots)
        job_ids.append(job.job_id())

    path = Path(path or f"jobs/{backend.name}_{time.strftime('%Y%m%d-%H%M%S')}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"provider": "ibm" if ibm else "iqm", "backend": backend.name, "shots": shots,
              "n_circuits": len(isa), "job_ids": job_ids}
    path.write_text(json.dumps(record, indent=1))
    print(f"{len(job_ids)} job(s) on {backend.name} saved to {path}")
    return path


def retrieve(path: str | Path, backend) -> list[Counts]:
    """Wait for the jobs saved by ``submit`` and return one counts dictionary per circuit, in submission order.
    ``backend`` is the device the jobs ran on (reconnect with ``ibm_backend``/``iqm_backend`` after a restart)."""
    record = json.loads(Path(path).read_text())
    if record["backend"] != backend.name:
        raise ValueError(f"{path} holds jobs on {record['backend']!r}, not {backend.name!r}")
    if record["provider"] == "ibm":
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = getattr(backend, "service", None) or QiskitRuntimeService()
    counts: list[Counts] = []
    for job_id in tqdm(record["job_ids"], desc="retrieving", unit="job"):
        if record["provider"] == "ibm":
            counts += [pub.join_data().get_counts() for pub in service.job(job_id).result()]
        else:
            result = backend.retrieve_job(job_id).result()
            counts += [result.get_counts(i) for i in range(len(result.results))]
    if len(counts) != record["n_circuits"]:
        raise RuntimeError(f"expected {record['n_circuits']} counts, got {len(counts)}")
    return counts
