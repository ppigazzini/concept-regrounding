"""Print what torch's vector maths gives on this CPU, to compare between machines.

    uv run python tools/vector_maths.py

Under the reproducible profile torch computes ``sqrt``, ``exp``, ``log`` and the other
functions of ATen's ``cpu/vml.h`` through MKL's vector maths, which picks its kernel by CPU
vendor. The line printed is the digest of torch's square roots of 131072 seeded float32
values and how many of them differ from the correctly rounded square root, NumPy's. No
training step calls these functions (``regrounding/train.py``).
"""

import hashlib

import numpy as np

from regrounding import reproducible

reproducible.pin()

import torch  # noqa: E402 - after the environment is pinned


def main() -> None:
    """Print the digest of the square roots and the count that is not correctly rounded."""
    values = torch.rand(131_072, generator=torch.Generator().manual_seed(0))
    roots = torch.sqrt(values).numpy()
    off = int((roots != np.sqrt(values.numpy())).sum())
    digest = hashlib.sha256(roots.tobytes()).hexdigest()[:16]
    print(f"torch.sqrt of {values.numel()} float32 values: digest {digest}, {off} rounded off")


if __name__ == "__main__":
    main()
