"""The ONNX Runtime surface this project depends on.

Only the CPU inference path is described: session construction, input
introspection and `run`. Training, IO binding and provider configuration beyond
a provider-name list are deliberately omitted.
"""

from collections.abc import Mapping, Sequence
from enum import IntEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

class GraphOptimizationLevel(IntEnum):
    ORT_DISABLE_ALL = 0
    ORT_ENABLE_BASIC = 1
    ORT_ENABLE_EXTENDED = 2
    ORT_ENABLE_ALL = 99

class NodeArg:
    name: str
    type: str
    shape: list[int | str | None]

class SessionOptions:
    def __init__(self) -> None: ...
    intra_op_num_threads: int
    inter_op_num_threads: int
    #: 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal.
    log_severity_level: int
    graph_optimization_level: GraphOptimizationLevel
    enable_profiling: bool

class InferenceSession:
    def __init__(
        self,
        path_or_bytes: str | bytes,
        sess_options: SessionOptions | None = ...,
        providers: Sequence[str] | None = ...,
        provider_options: Sequence[Mapping[str, Any]] | None = ...,
        **kwargs: Any,
    ) -> None: ...
    def get_inputs(self) -> list[NodeArg]: ...
    def get_outputs(self) -> list[NodeArg]: ...
    def get_providers(self) -> list[str]: ...
    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, NDArray[Any]],
        run_options: Any = ...,
    ) -> list[NDArray[np.float32]]: ...

def get_available_providers() -> list[str]: ...

__version__: str
