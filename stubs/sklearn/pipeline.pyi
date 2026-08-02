from typing import Any

from numpy.typing import NDArray

from sklearn.base import BaseEstimator

class Pipeline(BaseEstimator):
    def __init__(
        self,
        steps: list[tuple[str, BaseEstimator]],
        *,
        memory: str | None = ...,
        verbose: bool = ...,
    ) -> None: ...
    #: Fitted steps by name. Typed as BaseEstimator rather than a per-step
    #: mapping because the step names are only known at construction.
    named_steps: dict[str, BaseEstimator]
    steps: list[tuple[str, BaseEstimator]]
    def __getitem__(self, index: int | str) -> BaseEstimator: ...
    def transform(self, X: NDArray[Any]) -> NDArray[Any]: ...
