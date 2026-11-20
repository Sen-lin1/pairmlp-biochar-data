# -*- coding: utf-8 -*-
"""Minimal runtime compatibility module for the deployed Pair-MLP v2 model.

The serialized model records its class as ``optimize_pair_v2.PairV2Compiled``.
Only inference code is kept here, so the GUI package does not need the original
training scripts or CatBoost training environment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


BASE_FEATURES = [
    "Adsorbent",
    "Pyrolysis temperature",
    "Pyrolysis time",
    "C",
    "H",
    "O",
    "N",
    "(O+N)/C",
    "Ash",
    "H/C",
    "O/C",
    "N/C",
    "Surface area",
    "Pore volume",
    "Average pore size",
    "Pollutant",
    "Adsorption time",
    "Initial concentration",
    "Solution pH",
    "RPM",
    "Volume",
    "Adsorbent dosage",
    "Adsorption temperature",
    "Ion concentration",
    "Humic acid",
    "Wastewater type",
    "Adsorption type",
]

CATEGORICAL_FEATURES = [
    "Adsorbent",
    "Pollutant",
    "Wastewater type",
    "Adsorption type",
]

NUM = [feature for feature in BASE_FEATURES if feature not in CATEGORICAL_FEATURES]
LOG_FEATURES = [
    "Adsorption time",
    "Initial concentration",
    "Adsorbent dosage",
    "Surface area",
    "Pore volume",
]


def prepare_arrays(data: pd.DataFrame):
    """Validate raw features and reproduce the deployed base feature transform."""
    missing = [feature for feature in BASE_FEATURES if feature not in data.columns]
    if missing:
        raise ValueError("Missing model inputs: " + ", ".join(missing))

    numeric = data[NUM].to_numpy(dtype=float)
    categories = data[CATEGORICAL_FEATURES].to_numpy(dtype=object)

    if not np.isfinite(numeric).all() or pd.isna(categories).any():
        raise ValueError("Missing or non-finite input")

    initial = numeric[:, NUM.index("Initial concentration")]
    dosage = numeric[:, NUM.index("Adsorbent dosage")]
    contact = numeric[:, NUM.index("Adsorption time")]

    if np.any(dosage <= 0) or np.any(initial < 0) or np.any(contact < 0):
        raise ValueError("Invalid dose, concentration or contact time")

    capacity_limit = initial / dosage
    log_values = np.log1p(
        numeric[:, [NUM.index(feature) for feature in LOG_FEATURES]]
    )
    transformed = np.concatenate(
        [
            numeric,
            log_values,
            capacity_limit[:, None],
            np.log1p(capacity_limit)[:, None],
        ],
        axis=1,
    )
    return transformed, categories, capacity_limit, contact


def pair_labels(data: pd.DataFrame) -> np.ndarray:
    """Create the material-pollutant interaction category used during training."""
    return (
        data["Adsorbent"].astype(str).to_numpy()
        + " || "
        + data["Pollutant"].astype(str).to_numpy()
    )


def v2_arrays(data: pd.DataFrame, mode: str):
    """Reproduce the exact numeric and categorical arrays used by Pair-MLP v2."""
    numeric, base_categories, capacity_limit, contact = prepare_arrays(data)
    extras = []

    if mode in {"kinetic", "full"}:
        initial = numeric[:, NUM.index("Initial concentration")]
        log_initial = np.log1p(initial)
        log_contact = np.log1p(contact)

        for tau in (30.0, 120.0, 480.0):
            saturation = contact / (contact + tau)
            extras.extend(
                [
                    saturation[:, None],
                    (capacity_limit * saturation)[:, None],
                ]
            )

        extras.append((log_initial * log_contact)[:, None])

    if mode == "full":
        ph = numeric[:, NUM.index("Solution pH")]
        temperature = numeric[:, NUM.index("Adsorption temperature")]
        surface_area = numeric[:, NUM.index("Surface area")]
        pore_volume = numeric[:, NUM.index("Pore volume")]
        dosage = numeric[:, NUM.index("Adsorbent dosage")]

        extras.extend(
            [
                (ph - 7.0)[:, None],
                ((ph - 7.0) ** 2)[:, None],
                (1000.0 / (temperature + 273.15))[:, None],
                (surface_area * dosage)[:, None],
                (
                    1000.0
                    * pore_volume
                    / np.maximum(surface_area, 1e-9)
                )[:, None],
            ]
        )

    if extras:
        numeric = np.concatenate([numeric, *extras], axis=1)

    categories = np.column_stack([base_categories, pair_labels(data)])
    return numeric, categories, capacity_limit, contact


class PairV2Compiled:
    """Inference-only class matching the class stored in ``selected_model.pkl``."""

    def predict_prepared(
        self,
        numeric: np.ndarray,
        categories: np.ndarray,
        capacity_limit: np.ndarray,
        contact: np.ndarray,
    ) -> np.ndarray:
        hidden = numeric @ self.W1 + self.b1

        for column, lookup in enumerate(self.lookup):
            hidden += np.stack(
                [lookup.get(value, self.zero) for value in categories[:, column]]
            )

        hidden = np.tanh(hidden).reshape(len(numeric), self.n, self.h1)
        hidden = np.tanh(
            np.einsum("bsi,sij->bsj", hidden, self.W2, optimize=False) + self.b2
        )
        prediction = (
            np.einsum("bsj,sj->bs", hidden, self.W3, optimize=False) + self.b3
        ).mean(axis=1)

        return np.where(
            contact == 0,
            0,
            np.clip(prediction, 0, capacity_limit),
        )

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return self.predict_prepared(*v2_arrays(data, self.mode))
