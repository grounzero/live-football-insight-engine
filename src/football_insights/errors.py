"""Exception hierarchy.

The engine fails loudly on invalid source data rather than inventing observations.
Each error carries enough context to diagnose the offending match, period or frame.
"""

from __future__ import annotations


class FootballInsightsError(Exception):
    """Base class for every error raised by this package."""


class DataValidationError(FootballInsightsError):
    """Source data violates an expected schema, range or ordering guarantee."""


class OrientationError(DataValidationError):
    """Attacking direction could not be established with sufficient confidence.

    Raised when the evidence hierarchy in :mod:`football_insights.data.orientation`
    finds material disagreement between signals, or when a team's direction fails
    to flip between the first and second half.
    """


class SchemaVersionError(FootballInsightsError):
    """A model artifact was built against an incompatible feature schema."""


class ModelUnavailableError(FootballInsightsError):
    """Inference was requested while no predictor is loaded.

    Distinct from a legitimate low-confidence prediction: this is a service
    failure and must be reflected in readiness, never rendered as an insight.
    """


class InsufficientDataError(FootballInsightsError):
    """A rolling window does not contain enough valid frames to score."""
