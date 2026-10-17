"""Controller interface for the deterministic ALE building MPC driver."""

from __future__ import annotations

def fit_model(training_rows: list[dict]) -> dict:
    """Fit and return a JSON-serializable thermal model.

    Each training row contains ``t_zone``, ``t_outdoor``, ``solar``,
    ``occupancy``, ``hour_of_day``, ``flow_kg_s``, and ``next_t_zone``.
    """
    raise NotImplementedError


def predict_next_temperature(model: dict, observation: dict) -> float:
    """Predict the next 15-minute zone temperature."""
    raise NotImplementedError


def select_action(mode: str, observation: dict, forecast: list[dict], model: dict) -> float:
    """Return one of 0.0, 0.1, or 0.3 kg/s for the requested policy mode."""
    raise NotImplementedError
