"""Shared type aliases."""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

#: A JSON-shaped document: reports, manifests and serialised metadata.
#:
#: ``dict[str, object]`` would be more precise about immutability but makes every
#: consumer cast before indexing, which buys nothing for structures that are
#: written straight to JSON and read back untyped.
JsonDict: TypeAlias = dict[str, Any]

# ---------------------------------------------------------------- arrays
#
# Bare ``np.ndarray`` carries no dtype, so every element access downstream
# degrades to ``Any`` and a strict checker can no longer tell a probability
# array from a label array. These aliases name the dtypes this project actually
# moves around. They constrain dtype only — shape stays ``tuple[Any, ...]``,
# because Python's type system cannot express "(n, sequence_length, n_features)"
# and pretending otherwise would be worse than documenting it in the docstring.

#: Probabilities, metrics, timestamps and pitch coordinates in metres.
FloatArray: TypeAlias = npt.NDArray[np.float64]

#: Model-facing feature windows. float32 is the model input contract.
Float32Array: TypeAlias = npt.NDArray[np.float32]

#: Binary labels. int8 keeps large label arrays cheap and is the on-disk dtype.
LabelArray: TypeAlias = npt.NDArray[np.int8]

#: Boolean masks over frames, windows or episodes.
BoolArray: TypeAlias = npt.NDArray[np.bool_]

#: Frame numbers, team codes and episode identifiers. int64 in every reader and
#: in the processed ``.npz`` files, so widening this would misdescribe the data.
IntArray: TypeAlias = npt.NDArray[np.int64]

#: Period numbers. int16: periods run 1-5 but the array carries one entry per
#: frame, so the narrower dtype is worth keeping. All three readers agree on it.
Int16Array: TypeAlias = npt.NDArray[np.int16]
