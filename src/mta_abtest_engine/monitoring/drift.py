"""Feature drift with Evidently (ADR 0021).

Reference = what the champion was trained on (train scoring points); current =
the sessions scored on one day. Evidently picks a test per column (K-S /
chi-square on small samples, Wasserstein / Jensen-Shannon distance on large ones)
and reports the share of drifted columns. The HTML report is for people; the
parsed numbers feed the retraining triggers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

_VALUE_DRIFT = re.compile(
    r"ValueDrift\(column=(.+?),method=(.+?),threshold=([\d.eE+-]+)\)"
)


@dataclass(frozen=True)
class FeatureDrift:
    share_drifted: float
    drifted_columns: list[str]
    report_path: Path


def feature_drift(
    reference: pd.DataFrame, current: pd.DataFrame, report_path: Path
) -> FeatureDrift:
    # All our features are numeric: declare it, so Evidently does not guess that
    # low-cardinality counts are categories (mixed types break its tests).
    definition = DataDefinition(numerical_columns=list(reference.columns))
    # Evidently plots values against the index: features are indexed by
    # session_key (strings), which breaks its binning. Use a positional index.
    reference = reference.reset_index(drop=True)
    current = current.reset_index(drop=True)
    # Constant columns (e.g. a rare channel absent that day) make Evidently's
    # correlations divide by zero; harmless, so silenced locally.
    with np.errstate(invalid="ignore", divide="ignore"):
        snapshot = Report([DataDriftPreset()]).run(
            current_data=Dataset.from_pandas(current, data_definition=definition),
            reference_data=Dataset.from_pandas(reference, data_definition=definition),
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(report_path))
    share, drifted = 0.0, []
    for metric in snapshot.dict()["metrics"]:
        name = metric.get("metric_name", "")
        if name.startswith("DriftedColumnsCount"):
            share = float(metric["value"]["share"])
        match = _VALUE_DRIFT.match(name)
        if match:
            column, method, threshold = match.groups()
            value = float(metric["value"])
            # p-values drift when small; distances drift when large.
            is_drift = (
                value < float(threshold)
                if "p_value" in method
                else (value > float(threshold))
            )
            if is_drift:
                drifted.append(column)
    return FeatureDrift(share, sorted(drifted), report_path)
