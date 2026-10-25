"""Seeding and the runtime switches of the reproducible profile.

Seeding before any model or data is built, deterministic algorithms, cuDNN determinism
and no TF32 on a GPU, one thread and neither oneDNN nor NNPACK on the CPU. The environment
variables that must precede ``import torch`` live in `regrounding.reproducible`.
"""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU and CUDA) from one value."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - the legacy global some libraries still read
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_runtime(device: torch.device) -> None:
    """Make every operation deterministic on ``device``.

    oneDNN and NNPACK are switched off on the CPU because each picks its convolution
    kernel and blocking by CPU model, and torch hands NNPACK every batch of 16 scenes or
    more; the native convolution then runs as a matrix product on MKL, which
    `regrounding.reproducible` pins to one code path.
    """
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False  # ty: ignore[invalid-assignment] - a module flag
    torch.backends.nnpack.set_flags(False)
    torch.use_deterministic_algorithms(True)  # noqa: FBT003 - torch's own positional flag
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def resolve(name: str) -> torch.device:
    """Return the device ``name`` asks for; ``auto`` is CUDA when one is present."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
