from collections.abc import Mapping
from typing import Any, Literal

from sklearn.base import BaseEstimator

class LogisticRegression(BaseEstimator):
    def __init__(
        self,
        *,
        penalty: str | None = ...,
        tol: float = ...,
        C: float = ...,
        fit_intercept: bool = ...,
        class_weight: Mapping[int, float] | Literal["balanced"] | None = ...,
        random_state: int | None = ...,
        solver: str = ...,
        max_iter: int = ...,
        n_jobs: int | None = ...,
        **kwargs: Any,
    ) -> None: ...
