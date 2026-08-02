from collections.abc import Mapping
from typing import Any, Literal

from sklearn.base import BaseEstimator

class HistGradientBoostingClassifier(BaseEstimator):
    def __init__(
        self,
        *,
        loss: str = ...,
        learning_rate: float = ...,
        max_iter: int = ...,
        max_leaf_nodes: int | None = ...,
        max_depth: int | None = ...,
        min_samples_leaf: int = ...,
        l2_regularization: float = ...,
        max_bins: int = ...,
        # bool | "auto" at runtime; the source default of "auto" makes an
        # inferred signature reject the documented boolean form.
        early_stopping: bool | Literal["auto"] = ...,
        scoring: str | None = ...,
        validation_fraction: float | int | None = ...,
        n_iter_no_change: int = ...,
        tol: float = ...,
        verbose: int = ...,
        random_state: int | None = ...,
        class_weight: Mapping[int, float] | Literal["balanced"] | None = ...,
        **kwargs: Any,
    ) -> None: ...
