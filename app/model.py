"""Fixed-weight NumPy recurrent quality model.

The model is intentionally small and deterministic: an 8-feature input is fed
into a 6-unit tanh recurrence, and a logistic head produces a defect score in
[0, 1].  Weights are literals in this file (never trained at runtime), so the
same feature stream always produces the same score and replay is reproducible.

    h_t = tanh(W x_t + U h_{t-1} + b)
    s_t = sigmoid(v . h_t + bias)
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

FEATURES = 8
HIDDEN = 6
BIAS = -2.0
MODEL_NAME = "numpy-recurrent-quality-v1"

W = np.array(
    [
        [0.410573, 0.411135, 0.210190, 0.262583, 0.317971, 0.358658, 0.370678, 0.280554],
        [0.357649, 0.339296, 0.399021, 0.290851, 0.184828, 0.383735, 0.269163, 0.378466],
        [0.380832, 0.203111, 0.280703, 0.206491, 0.290660, 0.213682, 0.389741, 0.246931],
        [0.332559, 0.235901, 0.337008, 0.196565, 0.359264, 0.223598, 0.248125, 0.309143],
        [0.309183, 0.313313, 0.358202, 0.267252, 0.291824, 0.182342, 0.233519, 0.274470],
        [0.381197, 0.234586, 0.203886, 0.350520, 0.357593, 0.324402, 0.333108, 0.388305],
    ],
    dtype=np.float64,
)

U = np.array(
    [
        [0.191298, -0.066327, -0.066610, 0.008872, 0.057588, 0.064609],
        [0.003856, 0.133857, 0.031033, -0.075733, 0.072504, -0.017369],
        [-0.063038, -0.016246, 0.207482, -0.079501, -0.014653, 0.051152],
        [0.078273, 0.050816, -0.057594, 0.153030, 0.046338, -0.017287],
        [0.077705, -0.043145, -0.046298, 0.017006, 0.127853, -0.027394],
        [0.042943, 0.018424, -0.067582, -0.063995, 0.032168, 0.181731],
    ],
    dtype=np.float64,
)

B = np.array([-0.158028, -0.237025, -0.179058, -0.114584, -0.287960, -0.104146], dtype=np.float64)
V = np.array([0.765011, 0.478938, 0.705265, 0.772056, 0.580880, 0.574126], dtype=np.float64)


def initial_state() -> np.ndarray:
    return np.zeros(HIDDEN, dtype=np.float64)


def step(state: np.ndarray, features: np.ndarray | list[float]) -> np.ndarray:
    """Advance the recurrence by one frame."""
    x = np.asarray(features, dtype=np.float64)
    if x.shape != (FEATURES,):
        raise ValueError(f"feature vector must have {FEATURES} values, got {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("feature vector contains non-finite values")
    return np.tanh(W @ x + U @ state + B)


def score(state: np.ndarray) -> float:
    """Defect score for a hidden state (stable logistic)."""
    z = float(np.clip(V @ state + BIAS, -60.0, 60.0))
    return 1.0 / (1.0 + math.exp(-z))


def state_digest(state: np.ndarray) -> str:
    blob = np.round(np.asarray(state, dtype=np.float64), 6).astype("<f8").tobytes()
    return hashlib.sha256(blob).hexdigest()[:12]


def weight_digest() -> str:
    """Digest of the fixed weight literals, for audit and drift detection."""
    blob = b"".join(a.astype("<f8").tobytes() for a in (W, U, B, V))
    blob += np.float64(BIAS).astype("<f8").tobytes()
    return hashlib.sha256(blob).hexdigest()[:16]


def info() -> dict[str, Any]:
    return {
        "name": MODEL_NAME,
        "features": FEATURES,
        "hidden": HIDDEN,
        "bias": BIAS,
        "weight_digest": weight_digest(),
        "fixed_weights": True,
        "formula": "h_t = tanh(W x_t + U h_{t-1} + b); s_t = sigmoid(v . h_t + bias)",
    }
