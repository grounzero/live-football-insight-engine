"""Typed adapters over gaps in PyTorch's own type information.

PyTorch ships ``py.typed``, but a few of its public functions are declared with
unannotated parameters, so a strict checker sees ``(ndarray: Unknown) -> Tensor``
and reports every call site. The gap is in the library, not in this project, and
the alternative to an adapter is the same suppression repeated at six call
sites.

Each adapter below states the contract the project actually relies on — dtype
and memory sharing — which is information the real signature would not have
given us anyway.

If a future PyTorch release annotates these, delete this module and call
``torch`` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


def tensor_from(array: npt.NDArray[np.float32]) -> torch.Tensor:
    """Wrap a float32 NumPy array as a tensor **sharing its memory**.

    The caller must not mutate ``array`` afterwards, and must pass float32:
    ``torch.from_numpy`` preserves the source dtype, so a float64 array would
    silently produce a float64 tensor and fail against float32 model weights.

    Args:
        array: A contiguous float32 array.

    Returns:
        A tensor viewing the same buffer.
    """
    # torch.from_numpy is declared as (ndarray: Unknown) -> Tensor in torch 2.13.
    return torch.from_numpy(array)  # pyright: ignore[reportUnknownMemberType]


def seed_torch(seed: int) -> None:
    """Seed PyTorch's global RNG.

    Args:
        seed: The seed to apply.
    """
    # torch.manual_seed is declared as (seed: Unknown) -> Generator in torch 2.13.
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
