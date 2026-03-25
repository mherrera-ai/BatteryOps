"""Baseline predictive maintenance models."""

from pathlib import Path

__all__ = ["train_baselines"]


def train_baselines(
    processed_dir: Path | str = Path("data/processed"),
    artifact_dir: Path | str = Path("artifacts/demo"),
    allow_demo_fallback: bool = True,
) -> object:
    """Proxy import to keep `python -m batteryops.models.train` warning-free."""
    from batteryops.models.train import train_baselines as _train_baselines

    return _train_baselines(
        processed_dir=processed_dir,
        artifact_dir=artifact_dir,
        allow_demo_fallback=allow_demo_fallback,
    )
