"""The environment the reproducible profile needs, set before torch is imported.

A published count must come out the same on another x86 machine, and three things in a
CPU build of torch otherwise vary with the machine: ATen's vectorized kernels pick the
widest instruction set the CPU has, and a wider vector sums in a different order; MKL's
matrix products do the same unless told to use one code path; and more than one thread
splits a reduction differently. Each is pinned here, which is why this module imports
nothing that imports torch: ATen and MKL read these variables when they start. A fourth,
MKL's vector math, picks its kernel by CPU vendor whatever the code path set here;
`regrounding.train` keeps every training step off it.
"""

import os

#: ATen's portable kernels, MKL's conditional numerical reproducibility, one thread, and
#: cuBLAS's deterministic workspace for a run that is pointed at a GPU anyway.
ENVIRONMENT = {
    "ATEN_CPU_CAPABILITY": "default",
    "MKL_CBWR": "COMPATIBLE",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}


def pin(*, native_kernels: bool = False) -> None:
    """Set the reproducible environment; call before the first ``import torch``.

    ``native_kernels`` leaves ATen free to pick the widest kernels the CPU has, the one
    change a check makes to measure how far a kernel's rounding moves the counts; no study
    is written with it.
    """
    os.environ.update(ENVIRONMENT)
    if native_kernels:
        del os.environ["ATEN_CPU_CAPABILITY"]


def hide_gpus() -> None:
    """Hide every GPU from this process; call before torch first touches CUDA.

    A CPU run opens a CUDA context anyway when a GPU is present: torch's optimizers look
    up the current CUDA stream on every step. One context per worker process exhausts the
    GPU's memory before the cores run out, and the step then fails.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
