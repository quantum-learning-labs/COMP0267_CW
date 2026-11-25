"""Runtime setup, device selection and seeding."""
import importlib.metadata, random, subprocess, sys

import numpy as np, torch


def setup() -> None:
    """Prepare the runtime: on a GPU, replace Qiskit Aer by its GPU build of the same version. Run before importing Aer."""
    installed = {(d.metadata["Name"] or "").lower().replace("_", "-"): d.version for d in importlib.metadata.distributions()}
    has_gpu_aer = any(name.startswith("qiskit-aer-gpu") for name in installed)
    if torch.cuda.is_available() and "qiskit-aer" in installed and not has_gpu_aer:
        for args in (("uninstall", "-y", "qiskit-aer"), ("install", f"qiskit-aer-gpu=={installed['qiskit-aer']}")):
            subprocess.run([sys.executable, "-m", "pip", "-q", *args], check=True)
    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "none"
    print(f"torch {torch.__version__} | GPU: {gpu}")


def get_device() -> torch.device:
    """CUDA if a working GPU is present, else CPU (the quantum models use complex tensors, which MPS lacks)."""
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda") + 1
            return torch.device("cuda")
        except RuntimeError as err:
            print(f"CUDA is present but unusable ({err}); using the CPU.")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
