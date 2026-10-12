"""Persistence helpers for the sklearn predictor and PyTorch task encoder."""

from pathlib import Path
from typing import Union

import joblib
import torch

from gnn_encoder import TaskGraphEncoder
from task_predictors import TaskResourcePredictor

PathLike = Union[str, Path]


def save_predictor(predictor: TaskResourcePredictor, path: PathLike) -> None:
    """Save a fitted scikit-learn resource predictor with joblib."""
    if not predictor.is_fitted:
        raise ValueError("cannot save a predictor before it has been fitted")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(predictor, target)


def load_predictor(path: PathLike) -> TaskResourcePredictor:
    """Load and validate a predictor saved by :func:`save_predictor`."""
    predictor = joblib.load(Path(path))
    if not isinstance(predictor, TaskResourcePredictor) or not predictor.is_fitted:
        raise TypeError("file does not contain a fitted TaskResourcePredictor")
    return predictor


def save_gnn_encoder(encoder: TaskGraphEncoder, path: PathLike) -> None:
    """Save encoder configuration and weights using a safe tensor checkpoint."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_dim": encoder.input_dim,
            "hidden_dim": encoder.hidden_dim,
            "embedding_dim": encoder.embedding_dim,
            "output_dim": encoder.output_dim,
            "backend": encoder.backend,
            "state_dict": encoder.state_dict(),
        },
        target,
    )


def load_gnn_encoder(
    path: PathLike, *, device: torch.device = torch.device("cpu")
) -> TaskGraphEncoder:
    """Load a saved encoder onto the requested device."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if checkpoint.get("output_dim") != 4:
        raise ValueError("GNN checkpoint does not contain a four-resource prediction head")
    encoder = TaskGraphEncoder(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        embedding_dim=checkpoint["embedding_dim"],
        backend=checkpoint["backend"],
    )
    encoder.load_state_dict(checkpoint["state_dict"])
    return encoder.to(device)


__all__ = [
    "load_gnn_encoder",
    "load_predictor",
    "save_gnn_encoder",
    "save_predictor",
]
