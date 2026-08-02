from typing import Any

from numpy.typing import ArrayLike

def average_precision_score(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    average: str | None = ...,
    pos_label: int | str = ...,
    sample_weight: ArrayLike | None = ...,
) -> float: ...
def brier_score_loss(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    *,
    sample_weight: ArrayLike | None = ...,
    pos_label: int | str | None = ...,
    **kwargs: Any,
) -> float: ...
