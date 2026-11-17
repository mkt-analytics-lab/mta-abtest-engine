"""First-order Markov attribution over journey states (ADR 0016).

A journey is a path START -> s1 -> ... -> sn -> CONVERSION | NULL. Transition
probabilities are estimated from path counts; the conversion probability is the
probability of being absorbed in CONVERSION starting from START. The removal
effect of a state (or of every state of a channel) is the relative drop in that
probability when the state is removed (its traffic goes to NULL). Credits are
removal effects normalised to the observed conversions. Pure functions.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

START, CONVERSION, NULL = "(start)", "(conversion)", "(null)"


class PathCounts:
    """Unique paths with their weights; transitions precomputed once."""

    def __init__(self, paths: pd.DataFrame) -> None:
        """``paths``: columns path (tuple of states), converted (bool), n (count)."""
        self.paths = paths.reset_index(drop=True)
        states = sorted({s for path in self.paths["path"] for s in path})
        self.states = [START, *states, CONVERSION, NULL]
        self.index = {s: i for i, s in enumerate(self.states)}
        rows, cols = [], []
        k = len(self.states)
        for row, (path, converted) in enumerate(
            zip(self.paths["path"], self.paths["converted"], strict=True)
        ):
            full = [START, *path, CONVERSION if converted else NULL]
            for a, b in zip(full, full[1:], strict=False):
                rows.append(row)
                cols.append(self.index[a] * k + self.index[b])
        self.transitions = sparse.csr_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(len(self.paths), k * k)
        )

    def transition_counts(self, weights: np.ndarray | None = None) -> np.ndarray:
        w = self.paths["n"].to_numpy(float) if weights is None else weights
        k = len(self.states)
        return np.asarray(self.transitions.T @ w).reshape(k, k)


def conversion_probability(
    counts: np.ndarray, states: list[str], removed: Iterable[str] = ()
) -> float:
    """P(absorbed in CONVERSION | START), with ``removed`` states sent to NULL."""
    idx = {s: i for i, s in enumerate(states)}
    counts = counts.copy()
    for state in removed:
        i = idx[state]
        counts[i, :] = 0.0
        counts[i, idx[NULL]] = 1.0
    absorbing = [idx[CONVERSION], idx[NULL]]
    transient = [i for i in range(len(states)) if i not in absorbing]
    totals = counts[transient].sum(axis=1, keepdims=True)
    probs = np.divide(
        counts[transient],
        totals,
        out=np.zeros_like(counts[transient]),
        where=totals > 0,
    )
    Q = probs[:, transient]
    R = probs[:, absorbing]
    B = np.linalg.solve(np.eye(len(transient)) - Q, R)
    return float(B[transient.index(idx[START]), 0])


def removal_effects(
    counts: np.ndarray, states: list[str], groups: dict[str, list[str]]
) -> pd.Series:
    """Relative drop in conversion probability when each group is removed."""
    base = conversion_probability(counts, states)
    if base <= 0:  # no conversions (e.g. a bootstrap resample): nothing to credit
        return pd.Series(0.0, index=list(groups), dtype=float)
    effects = {
        name: 1 - conversion_probability(counts, states, members) / base
        for name, members in groups.items()
    }
    return pd.Series(effects, dtype=float)


def credits(effects: pd.Series, total: float) -> pd.Series:
    """Normalise removal effects so that credits add up to ``total``."""
    positive = effects.clip(lower=0)
    if positive.sum() == 0:
        return positive * 0
    return positive / positive.sum() * total


def state_groups(states: list[str]) -> dict[str, list[str]]:
    journey = [s for s in states if s not in (START, CONVERSION, NULL)]
    return {s: [s] for s in journey}


def channel_groups(states: list[str]) -> dict[str, list[str]]:
    """States are '<channel>:<stage>': group them by channel."""
    groups: dict[str, list[str]] = {}
    for s in states:
        if s not in (START, CONVERSION, NULL):
            groups.setdefault(s.split(":")[0], []).append(s)
    return groups
