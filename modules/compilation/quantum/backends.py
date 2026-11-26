import torch, os, yaml
from tqdm import tqdm
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
import qiskit_ibm_runtime.fake_provider as fake_provider
from qiskit.transpiler import CouplingMap
from itertools import batched

from iqm.qiskit_iqm import IQMProvider
from iqm.qubit_selector.qubit_selector import CostEvaluator
from iqm.qubit_selector.qiskit_utils import perform_backend_transpilation

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

from modules.utils.general import gen_id


DEFAULT_PROFILE = {'ibm': 'miami', 'iqm': 'garnet'}

def fake_backend(name, provider='ibm'):
    if provider == 'ibm':
        classes = {c.lower(): c for c in dir(fake_provider) if c.startswith('Fake')}
        profile = name.lower().replace('fake_', '').replace('fake', '')
        aliases = {'miami': 'fakemontrealv2'}
        class_name = classes.get(aliases.get(profile, 'fake' + profile))
        if class_name is None:
            choices = sorted(classes.values())
            raise ValueError(f"IBM profile {name!r} is unavailable; choose from {choices}")
        return getattr(fake_provider, class_name)()
    if provider == 'iqm':
        return iqm_fake_backend(name)
    raise ValueError(f"emulate.provider {provider!r}: choose from {list(DEFAULT_PROFILE)}")

def iqm_fake_backend(name):
    # IQM's own snapshots (qiskit extra of iqm-client): native gates r (prx) and cz, noise model built from the error profile
    import importlib
    name = name.lower().replace('iqm', '').replace('fake', '').strip('_')
    devices = ['adonis', 'aphrodite', 'apollo', 'garnet']   # deneb routes through a resonator (move gates): not supported
    if name not in devices:
        raise ValueError(f"IQM profile {name!r}: choose from {devices}")
    module = importlib.import_module(f'iqm.qiskit_iqm.fake_backends.fake_{name}')
    return getattr(module, f'IQMFake{name.capitalize()}')()

def noise_model(backend, provider):
    return backend.noise_model if provider == 'iqm' else NoiseModel.from_backend(backend)
class BackendManager:
    def __init__(self, provider, profile, method='statevector', hardware_profile=None, qlimit=None, noisy=False):
        self.provider = provider
        self.profile = profile
        self.method = method
        self.hardware_profile = hardware_profile
        self.qlimit = qlimit
        self.noisy = noisy
        self.coupling_map = None
        self.basis_gates = None

        if self.provider not in DEFAULT_PROFILE:
            raise ValueError(f"emulate.provider {self.provider!r}: choose from {list(DEFAULT_PROFILE)}")
        self.hardware_profile = hardware_profile or fake_backend(profile, provider)
        self._backend = None
        self._setup_backend()

    @property
    def backend(self):
        return self._backend

    def _setup_backend(self):
        devices = AerSimulator().available_devices()
        qdev = 'GPU' if 'GPU' in devices and torch.cuda.is_available() else 'CPU'
        
        backend_options = {"device": qdev, "method": self.method}

        # Ensures not using more qubits than necessary
        edges = self.hardware_profile.coupling_map.get_edges()
        if self.qlimit is not None:
            filtered_edges = [edge for edge in edges if edge[0] < self.qlimit and edge[1] < self.qlimit]
            self.coupling_map = CouplingMap(filtered_edges)
            if not self.coupling_map.is_connected():
                print(f" [Warning] the first {self.qlimit} qubits of {self.hardware_profile.name} are not connected")
        else:
            self.coupling_map = self.hardware_profile.coupling_map
        self.basis_gates = getattr(self.hardware_profile, 'basis_gates', None) or list(self.hardware_profile.target.operation_names)

        if self.noisy:
            noise = noise_model(self.hardware_profile, self.provider)
            backend_options["noise_model"] = noise

        print(f" Simulation: {self.provider} profile {self.hardware_profile.name} ({self.qlimit} qubits), Device {qdev}, Method {self.method}, Noise {'Enabled' if self.noisy else 'Disabled'}")
        self._backend = AerSimulator(method=self.method, device=qdev)
        self._backend.set_options(**backend_options)

    def transpile(self, circuits, batch_size=128, optimization_level=1):
        if isinstance(circuits, tuple) and len(circuits) == 2:
            return tuple(
                self.transpile(side, batch_size=batch_size, optimization_level=optimization_level)
                for side in circuits
            )

        transpiled_circs = []
        chunks = [circuits[i:i + batch_size] for i in range(0, len(circuits), batch_size)]
        device = {'backend': self.hardware_profile}
        for chunk in tqdm(chunks, desc="Transpiling", unit="batch"):
            transpiled_chunk = transpile(
                chunk, 
                **device,
                optimization_level=optimization_level,
                num_processes=4,
            )
            transpiled_circs.extend(transpiled_chunk)
        return transpiled_circs

class IQMBackendManager(BackendManager):
    def __init__(self,profile, method='statevector', hardware_profile=None, qlimit=None):
        super().__init__(provider="IQM", profile=profile, method=method, hardware_profile=hardware_profile, qlimit=qlimit)

    def _setup_backend(self):
        iqm_server = "https://resonance.iqm.tech/" 

        provider = IQMProvider(iqm_server, quantum_computer = self.hardware_profile)
        self._backend = provider.get_backend()

        self.coupling_map = self._backend.coupling_map
        self.basis_gates = self._backend.operation_names

    def transpile(self, circuits, optimization_level=1):
        transpiled_circs = []

        for circuit in circuits:
            layouts, _ = CostEvaluator(backend=self._backend, quantum_circuit=circuit).get_top_layouts(num_layouts=1)

            transp_circuit = perform_backend_transpilation([circuit],
                                                            self._backend,
                                                            layouts[0],
                                                            self.coupling_map.reduce(mapping=layouts[0]),
                                                            qiskit_optim_level = optimization_level
                                                            )
            transpiled_circs.append(transp_circuit)
        return transpiled_circs
    
class IBMBackendManager(BackendManager):
    def __init__(self,profile, method='statevector', hardware_profile=None, qlimit=None):
            super().__init__(provider="IBM", profile=profile, method=method, hardware_profile=hardware_profile, qlimit=qlimit)
    
    
    def _setup_backend(self):
        QiskitRuntimeService.save_account(token=os.getenv('IBM_TOKEN'), instance=os.getenv('IBM_INSTANCE'), overwrite=True, set_as_default=True)
        service = QiskitRuntimeService()
        self._backend = service.backend(self.hardware_profile)
        self.coupling_map = self._backend.coupling_map
        self.basis_gates = self._backend.basis_gates
        if "xslow" in self.basis_gates:
            self.basis_gates.remove('xslow')


class ProviderManager:
    def __init__(self, backend_manager):
        self.backend_manager = backend_manager

    def submit_job(self, qc_array, shots):
        sampler = Sampler(self.backend)
        result = sampler.run(qc_array, shots=shots)
        job_id = result.job_id()

        self._save_job(qc_array, shots, job_id)

    def _save_job(self, shots, job_id):

        job_info = {
            "backend": self.backend_manager.provider,
            "job_id": job_id,
            "shots": shots,
        }

        save_filename = f"{self.backend_name}__{gen_id(self.config)}"
        print(f"Saved job information to {save_filename}")
        with open(f"jobs/{save_filename}.yaml", "w") as file:
            yaml.dump(job_info, file)

        return f"jobs/{save_filename}.yaml"

    def _get_results(self, job_id):
        job = self.backend_manager.backend().retrieve_job(job_id)
        return job.result().get_counts()

    def _get_counts(self, results, idx):
        return results[idx]
        
    def retrieve_job(self, job_info):
        
        results = self._get_results(job_info["job_id"])

        counts_array = []
        for idx in tqdm(range(len(results)), desc="Retrieving jobs", unit="job"):
            counts_array.append(self._get_counts(results, idx))

        return counts_array

class IQMProviderManager(ProviderManager):
    backend_name = "IQM"
    def __init__(self, config, backend_manager):
        super().__init__(config, backend_manager)

    def _get_results(self, job_ids):
        results = []
        for slice_id in job_ids:
            job = self.backend_manager.backend().retrieve_job(slice_id)
            results = results + job.result().get_counts()
        return results

    def submit_job(self, qc_array, shots):

        qc_arrays = batched(qc_array, 100) # IQM only allows 100 circuits per job
        job_ids = []
        for qc_slice in qc_arrays:
            result = self.backend.run(qc_slice, shots=shots)
            job_ids.append(result.job_id())

        self._save_job(qc_array, shots, job_ids)

class IBMProviderManager(ProviderManager):
    backend_name = "IBM"
    def __init__(self, config, backend_manager):
        super().__init__(config, backend_manager)


    def _get_results(self, job_id):
        job = self.backend_manager.service.job(job_id)
        return job.result()

    def _get_counts(self, results, idx):
        result = results[idx].join_data()
        return result.get_counts()


