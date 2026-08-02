# Local partial type stubs

`scikit-learn` and `onnxruntime` ship no `py.typed` marker and no stubs, and
neither has a maintained stub distribution on PyPI. Without stubs a strict
checker infers their public API from source, which produces two bad outcomes:

- `BaseEstimator` appears to have no `fit`, `predict_proba` or `named_steps`,
  because scikit-learn attaches them via mixins and `__init__`-time attribute
  assignment;
- keyword arguments are typed from their *default value* rather than their
  accepted domain, so `HistGradientBoostingClassifier(early_stopping=True)` is
  reported as an error purely because the default happens to be `"auto"`.

These stubs are **deliberately partial**. They describe only the API surface
this project actually uses, which keeps them small enough to stay correct. They
are not a general-purpose stub distribution and should not be published as one.

## Scope

| Module | Covered |
| --- | --- |
| `sklearn.base` | `BaseEstimator` (as the protocol this project relies on) |
| `sklearn.ensemble` | `HistGradientBoostingClassifier` |
| `sklearn.linear_model` | `LogisticRegression` |
| `sklearn.pipeline` | `Pipeline`, `named_steps` |
| `sklearn.preprocessing` | `StandardScaler` |
| `sklearn.metrics` | `average_precision_score`, `brier_score_loss` |
| `onnxruntime` | `InferenceSession`, `SessionOptions`, `GraphOptimizationLevel`, `NodeArg` |

## Maintenance

If a new scikit-learn or onnxruntime symbol is needed, add it here rather than
suppressing the diagnostic at the call site. If either project ever ships
`py.typed`, delete the corresponding directory and remove `stubPath` from
`pyrightconfig.json`, since the real types are always preferable to these.

The stubs are consulted by Pyright (`stubPath`) and by mypy (`mypy_path`), so
both checkers see the same boundary.
