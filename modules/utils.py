"""Device selection and seeding."""
import random

import numpy as np
import torch


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
