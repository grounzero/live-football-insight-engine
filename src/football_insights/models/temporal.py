"""Compact temporal model in PyTorch.

A single-layer GRU over the observation window, followed by a small head. The
architecture is deliberately modest: with roughly 200 positive episodes across
three matches, a larger model would memorise the training matches and tell us
nothing about the held-out one. The interesting question is not whether a big
network can fit this data — it can — but whether *any* sequence model beats four
lines of aggregate statistics on it. That question needs a model small enough
for the answer to be meaningful.

Everything is CPU-first. Training a model this size on 20,000 windows takes
under a minute on a laptop, and inference must run inside a live latency budget
where a GPU round trip would cost more than the forward pass.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from football_insights.features.spec import DEFAULT_FEATURE_SPEC, FeatureSpec
from football_insights.models.base import ModelMetadata, validate_batch
from football_insights.types import JsonDict

if TYPE_CHECKING:
    from collections.abc import Iterator

    from football_insights.config import ModelSettings


def seed_everything(seed: int) -> None:
    """Seed every RNG the training path touches.

    Full bit-for-bit reproducibility is not guaranteed across BLAS versions or
    hardware, which is why the reference run's exact metrics are recorded
    alongside its environment rather than presented as a target to hit.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(mode=True, warn_only=True)


def select_device(prefer_gpu: bool = False) -> torch.device:
    """Pick a device, defaulting to CPU.

    Args:
        prefer_gpu: Use CUDA or MPS when available.

    Returns:
        The chosen device.
    """
    if prefer_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


class GRUClassifier(nn.Module):
    """GRU encoder with a small classification head.

    The final hidden state summarises the window; a two-layer head maps it to a
    single logit. Outputs are logits rather than probabilities so training can
    use the numerically stable ``BCEWithLogitsLoss``, and so the ONNX graph
    stays free of a sigmoid whose placement would otherwise have to match
    between runtimes.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 48,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        """Build the network.

        Args:
            n_features: Features per timestep.
            hidden_size: GRU hidden width.
            num_layers: Number of stacked GRU layers.
            dropout: Dropout applied to the head, and between GRU layers when
                there is more than one.
        """
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """Score a batch.

        Args:
            windows: Tensor ``(batch, sequence_length, n_features)``.

        Returns:
            Logits of shape ``(batch, 1)``.
        """
        encoded, _ = self.gru(windows)
        logits: torch.Tensor = self.head(encoded[:, -1, :])
        return logits


@dataclass(frozen=True, slots=True)
class Standardiser:
    """Per-feature standardisation fitted on the training split.

    Stored with the model and applied identically at serving time; fitting it
    on anything but the training folds would leak the held-out distribution.
    """

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, windows: np.ndarray) -> Standardiser:
        """Fit over the batch and time axes."""
        flat = windows.reshape(-1, windows.shape[-1])
        mean = flat.mean(axis=0)
        scale = flat.std(axis=0)
        scale[scale < 1e-6] = 1.0
        return cls(mean=mean.astype(np.float32), scale=scale.astype(np.float32))

    def apply(self, windows: np.ndarray) -> np.ndarray:
        """Standardise a batch."""
        return np.asarray((windows - self.mean) / self.scale, dtype=np.float32)


class TemporalPredictor:
    """A trained :class:`GRUClassifier` behind the :class:`Predictor` interface."""

    def __init__(
        self,
        model: GRUClassifier,
        standardiser: Standardiser,
        metadata: ModelMetadata,
        device: torch.device | None = None,
        spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
    ) -> None:
        """Wrap a trained network.

        Args:
            model: The trained network.
            standardiser: Fitted standardiser.
            metadata: Model metadata.
            device: Inference device; CPU when omitted.
            spec: Feature schema.
        """
        self._model = model.eval()
        self._standardiser = standardiser
        self._metadata = metadata
        self._device = device or torch.device("cpu")
        self._spec = spec
        self._model.to(self._device)

    @property
    def metadata(self) -> ModelMetadata:
        """Identity and provenance."""
        return self._metadata

    @property
    def module(self) -> GRUClassifier:
        """The underlying network, for export."""
        return self._model

    @property
    def standardiser(self) -> Standardiser:
        """The fitted standardiser."""
        return self._standardiser

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        """Score a batch of windows.

        Args:
            windows: Array ``(n, sequence_length, n_features)``.

        Returns:
            Probabilities of shape ``(n,)``.
        """
        batch = validate_batch(windows, self._metadata)
        standardised = self._standardiser.apply(batch)
        with torch.no_grad():
            tensor = torch.from_numpy(standardised).to(self._device)
            logits = self._model(tensor)
            return torch.sigmoid(logits).squeeze(-1).cpu().numpy().astype(np.float64)

    def save(self, path: Path) -> None:
        """Persist weights, standardiser and metadata together."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "mean": self._standardiser.mean,
                "scale": self._standardiser.scale,
                "metadata": self._metadata.to_dict(),
                "hidden_size": self._model.gru.hidden_size,
                "num_layers": self._model.gru.num_layers,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: Path,
        device: torch.device | None = None,
        spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
    ) -> TemporalPredictor:
        """Load a persisted model, refusing a mismatched feature schema.

        Args:
            path: Artifact path.
            device: Inference device.
            spec: Feature schema the running code produces.

        Returns:
            The loaded predictor.

        Raises:
            SchemaVersionError: If the schemas disagree.
        """
        payload = torch.load(path, map_location="cpu", weights_only=False)
        raw = payload["metadata"]
        raw["training_matches"] = tuple(raw.get("training_matches", ()))
        metadata = ModelMetadata(**raw)
        metadata.require_schema(spec.schema_hash)
        model = GRUClassifier(
            n_features=metadata.n_features,
            hidden_size=int(payload["hidden_size"]),
            num_layers=int(payload["num_layers"]),
        )
        model.load_state_dict(payload["state_dict"])
        return cls(
            model,
            Standardiser(mean=payload["mean"], scale=payload["scale"]),
            metadata,
            device,
            spec,
        )


def _batches(
    n: int, batch_size: int, generator: np.random.Generator, shuffle: bool
) -> Iterator[np.ndarray]:
    """Yield index batches."""
    order = generator.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


@dataclass(slots=True)
class TrainingHistory:
    """Per-epoch record, kept for the run report."""

    train_loss: list[float]
    val_loss: list[float]
    val_pr_auc: list[float]
    best_epoch: int
    stopped_early: bool

    def to_dict(self) -> JsonDict:
        """Serialisable form."""
        return {
            "epochs_run": len(self.train_loss),
            "best_epoch": self.best_epoch,
            "stopped_early": self.stopped_early,
            "train_loss": [round(v, 5) for v in self.train_loss],
            "val_loss": [round(v, 5) for v in self.val_loss],
            "val_pr_auc": [round(v, 5) for v in self.val_pr_auc],
        }


def train_temporal(
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    val_windows: np.ndarray,
    val_labels: np.ndarray,
    settings: ModelSettings,
    *,
    spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
    training_matches: tuple[str, ...] = (),
    dataset_fingerprint: str | None = None,
    config_fingerprint: str | None = None,
    prefer_gpu: bool = False,
) -> tuple[TemporalPredictor, TrainingHistory]:
    """Train the temporal model with early stopping on validation PR-AUC.

    PR-AUC rather than loss or accuracy: with a 6% positive rate, loss improves
    steadily while ranking quality plateaus, and accuracy is uninformative.

    Args:
        train_windows: Training windows.
        train_labels: Training labels.
        val_windows: Validation windows.
        val_labels: Validation labels.
        settings: Model hyperparameters.
        spec: Feature schema.
        training_matches: Matches used, recorded in metadata.
        dataset_fingerprint: Dataset hash, recorded in metadata.
        config_fingerprint: Config hash, recorded in metadata.
        prefer_gpu: Allow CUDA or MPS if present.

    Returns:
        The best predictor by validation PR-AUC, and the training history.
    """
    from sklearn.metrics import average_precision_score

    seed_everything(settings.seed)
    device = select_device(prefer_gpu)
    standardiser = Standardiser.fit(train_windows)

    x_train = torch.from_numpy(standardiser.apply(train_windows))
    y_train = torch.from_numpy(train_labels.astype(np.float32)).unsqueeze(1)
    x_val = torch.from_numpy(standardiser.apply(val_windows)).to(device)
    y_val = val_labels.astype(np.int8)

    model = GRUClassifier(
        n_features=train_windows.shape[2],
        hidden_size=settings.hidden_size,
        num_layers=settings.num_layers,
        dropout=settings.dropout,
    ).to(device)

    positives = max(int(train_labels.sum()), 1)
    weight = settings.positive_class_weight or (len(train_labels) - positives) / positives
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], device=device))
    optimiser = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=1e-4)

    generator = np.random.default_rng(settings.seed)
    history = TrainingHistory([], [], [], best_epoch=0, stopped_early=False)
    best_score = -np.inf
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    since_improvement = 0

    for epoch in range(settings.max_epochs):
        model.train()
        losses: list[float] = []
        for index in _batches(len(y_train), settings.batch_size, generator, shuffle=True):
            batch_x = x_train[index].to(device)
            batch_y = y_train[index].to(device)
            optimiser.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            logits = model(x_val)
            val_loss = float(
                criterion(
                    logits, torch.from_numpy(y_val.astype(np.float32)).unsqueeze(1).to(device)
                )
            )
            probabilities = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        score = (
            float(average_precision_score(y_val, probabilities))
            if 0 < y_val.sum() < len(y_val)
            else 0.0
        )

        history.train_loss.append(float(np.mean(losses)))
        history.val_loss.append(val_loss)
        history.val_pr_auc.append(score)

        if score > best_score + 1e-5:
            best_score = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            history.best_epoch = epoch
            since_improvement = 0
        else:
            since_improvement += 1
            if since_improvement >= settings.early_stopping_patience:
                history.stopped_early = True
                break

    model.load_state_dict(best_state)
    metadata = ModelMetadata.now(
        name="gru-temporal",
        version="1.0.0",
        kind="gru",
        is_ml=True,
        feature_schema_hash=spec.schema_hash,
        sequence_length=int(train_windows.shape[1]),
        n_features=int(train_windows.shape[2]),
        training_matches=training_matches,
        dataset_fingerprint=dataset_fingerprint,
        config_fingerprint=config_fingerprint,
        metrics={"val_pr_auc": round(best_score, 5)},
        notes=(
            f"GRU hidden={settings.hidden_size} layers={settings.num_layers} "
            f"dropout={settings.dropout}, early stopping on validation PR-AUC"
        ),
    )
    return TemporalPredictor(model, standardiser, metadata, device, spec), history
