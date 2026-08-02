from typing import Any

import numpy as np
from numpy.typing import NDArray

from sklearn.base import BaseEstimator

class StandardScaler(BaseEstimator):
    def __init__(
        self, *, copy: bool = ..., with_mean: bool = ..., with_std: bool = ...
    ) -> None: ...
    def transform(self, X: NDArray[Any]) -> NDArray[np.float64]: ...
    def fit_transform(self, X: NDArray[Any], y: NDArray[Any] | None = ...) -> NDArray[np.float64]: ...
    mean_: NDArray[np.float64]
    scale_: NDArray[np.float64]
