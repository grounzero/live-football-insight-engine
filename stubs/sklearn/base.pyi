"""The estimator surface this project depends on.

scikit-learn spreads `fit`/`predict_proba` across mixins and sets learned
attributes such as `coef_` during `fit`. Declaring them on `BaseEstimator` is a
simplification, but it is the contract every estimator used here honours, and it
is checked at runtime by the tests that fit and score these models.
"""

from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

class BaseEstimator:
    def fit(self, X: NDArray[Any], y: NDArray[Any], **kwargs: Any) -> Self: ...
    def predict(self, X: NDArray[Any]) -> NDArray[np.int64]: ...
    def predict_proba(self, X: NDArray[Any]) -> NDArray[np.float64]: ...
    def get_params(self, deep: bool = True) -> dict[str, Any]: ...
    def set_params(self, **params: Any) -> Self: ...
    #: Set by linear models during `fit`; shape (n_classes, n_features).
    coef_: NDArray[np.float64]
    intercept_: NDArray[np.float64]
