"""The six metrics (Section V) and the formal quantities of Section VI.

Every class-fraction metric is computed from a partition dict
{unit_key: Prov}; event-fraction metrics take the event log.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .model import Event, Prov, System, counts

# --------------------------------------------------------------------------
# Class-fraction metrics
# --------------------------------------------------------------------------

def class_fraction(partition: Dict[tuple, Prov], cls: Prov) -> float:
    n = len(partition)
    return counts(partition)[cls] / n if n else 0.0


def drr(partition: Dict[tuple, Prov]) -> float:
    """Deliberate Reuse Ratio |U_D| / |U|."""
    return class_fraction(partition, Prov.D)


def drr_size(partition: Dict[tuple, Prov], reused_component_size: int) -> float:
    """Size-weighted variant comparable with the classical reuse level, Eq. (2)."""
    non_d = len(partition) - counts(partition)[Prov.D]
    denom = reused_component_size + non_d
    return reused_component_size / denom if denom else 0.0


def lrd_true(partition: Dict[tuple, Prov]) -> float:
    """LRD* = |U_L| / |U|."""
    return class_fraction(partition, Prov.L)


def lrd_typicality(
    partition: Dict[tuple, Prov],
    scores: Dict[tuple, float],
    tau: float,
) -> float:
    """Typicality estimator, Eq. (3): fraction of non-D units with token-normalised
    score <= tau. `scores` maps unit key -> s(u)."""
    n = len(partition)
    if not n:
        return 0.0
    flagged = sum(
        1 for k, c in partition.items() if c != Prov.D and k in scores and scores[k] <= tau
    )
    return flagged / n


def lrd_typicality_expectation(
    profile: Dict[Prov, float], recall: Dict[Prov, float], fp_human: float
) -> float:
    """Eq. (4): E[LRD_typ] = r_L p_L + r_P p_P + r_G p_G + f_H p_H."""
    return (
        recall.get(Prov.L, 0) * profile[Prov.L]
        + recall.get(Prov.P, 0) * profile[Prov.P]
        + recall.get(Prov.G, 0) * profile[Prov.G]
        + fp_human * profile[Prov.H]
    )


def gci_true(partition: Dict[tuple, Prov]) -> float:
    """GCI* = |U_G| / |U| (on a true partition)."""
    return class_fraction(partition, Prov.G)


gci_estimate = gci_true  # on an operational partition the same formula gives GCI-hat


# --------------------------------------------------------------------------
# Event-fraction metrics
# --------------------------------------------------------------------------

def _names(prompt: str, pattern_set: Set[str]) -> bool:
    p = prompt.lower()
    return any(n.lower() in p for n in pattern_set)


def ppc(events: Sequence[Event], pattern_set: Set[str]) -> float:
    """Event-weighted Prompted Pattern Coverage, Eq. (5)."""
    if not events:
        return 0.0
    return sum(1 for e in events if _names(e.prompt, pattern_set)) / len(events)


def ppc_units(events: Sequence[Event], pattern_set: Set[str]) -> float:
    """Unit-weighted PPC_U, Eq. (5)."""
    total = sum(len(e.units) for e in events)
    if not total:
        return 0.0
    return sum(len(e.units) for e in events if _names(e.prompt, pattern_set)) / total


def ppa(events: Sequence[Event], reconstructed: Set[str]) -> Tuple[float, float, float]:
    """Returns (PLC, PRF, PPA), Eqs. (6)-(8). `reconstructed` = ids of stored
    events whose prompt replays within the similarity threshold."""
    if not events:
        return 0.0, 0.0, 0.0
    stored = [e for e in events if e.prompt_stored]
    plc = len(stored) / len(events)
    if not stored:
        return plc, 0.0, 0.0
    rec = sum(1 for e in stored if e.event_id in reconstructed)
    prf = rec / len(stored)
    return plc, prf, plc * prf


# --------------------------------------------------------------------------
# Composite HRS
# --------------------------------------------------------------------------

def component_vector(drr_v, lrd_v, ppc_v, ppa_v, gci_v) -> np.ndarray:
    """v(U) = (DRR, 1-LRD, PPC, PPA, 1-GCI)."""
    return np.array([drr_v, 1 - lrd_v, ppc_v, ppa_v, 1 - gci_v], dtype=float)


def hrs(v: np.ndarray, beta: Sequence[float]) -> float:
    beta = np.asarray(beta, dtype=float)
    if beta.shape != (5,) or (beta < 0).any() or not math.isclose(beta.sum(), 1.0, abs_tol=1e-9):
        raise ValueError("beta must be a probability vector of length 5")
    return float(beta @ v)


def hrs_separated(v: np.ndarray, beta_unit: Sequence[float], beta_event: Sequence[float], gamma: float) -> float:
    """HRS_sep, Eq. (10): gamma * HRS_U + (1-gamma) * HRS_E."""
    bu = np.asarray(beta_unit, float); be = np.asarray(beta_event, float)
    if not (math.isclose(bu.sum(), 1) and math.isclose(be.sum(), 1) and 0 <= gamma <= 1):
        raise ValueError("weights must sum to one and gamma in [0,1]")
    hrs_u = bu @ v[[0, 1, 4]]
    hrs_e = be @ v[[2, 3]]
    return float(gamma * hrs_u + (1 - gamma) * hrs_e)


def hrs_lipschitz_bound(v: np.ndarray, beta: Sequence[float], beta2: Sequence[float]) -> float:
    """Theorem 7: |HRS(b) - HRS(b')| <= 0.5 * ||b - b'||_1 * range(v)."""
    d = np.abs(np.asarray(beta, float) - np.asarray(beta2, float)).sum()
    return 0.5 * d * (v.max() - v.min())


def calibrate_weights(X: np.ndarray, h: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Least-squares weights on the simplex (Theorem 6).

    Returns (beta, unique) where `unique` is the exact condition
    rank([X; 1^T]) == 5. Solved by projected gradient descent, which converges
    to the global minimiser because the problem is convex.
    """
    X = np.asarray(X, float); h = np.asarray(h, float)
    if X.shape[1] != 5:
        raise ValueError("X must have five columns")
    unique = np.linalg.matrix_rank(np.vstack([X, np.ones(5)])) == 5
    beta = np.full(5, 0.2)
    L = 2 * np.linalg.norm(X, 2) ** 2 + 1e-12
    for _ in range(20000):
        grad = 2 * X.T @ (X @ beta - h)
        beta = _project_simplex(beta - grad / L)
    return beta, bool(unique)


def _project_simplex(y: np.ndarray) -> np.ndarray:
    u = np.sort(y)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(y) + 1) > (css - 1))[0][-1]
    lam = (css[rho] - 1) / (rho + 1)
    return np.maximum(y - lam, 0)


# --------------------------------------------------------------------------
# Information loss (Theorem 4)
# --------------------------------------------------------------------------

def _H(p: Iterable[float]) -> float:
    return -sum(x * math.log2(x) for x in p if x > 0)


def information_loss(profile: Dict[Prov, float]) -> Tuple[float, float]:
    """Returns (loss, ceiling): L(U) = H(X) - H(Y) = (1-p_D) H(X|Y=0), and 2(1-p_D)."""
    pD = profile[Prov.D]
    non_d = [profile[c] for c in (Prov.L, Prov.P, Prov.G, Prov.H)]
    rest = sum(non_d)
    if rest <= 0:
        return 0.0, 0.0
    cond = _H([x / rest for x in non_d])
    return abs(rest * cond), 2 * rest


# --------------------------------------------------------------------------
# GCI robustness (Theorem 5)
# --------------------------------------------------------------------------

def gci_robustness_F(min_distances: Dict[tuple, float], n_units: int, theta: float, delta: float) -> float:
    """F(delta): fraction of all units whose min swap distance lies in
    [theta - delta, theta + delta]. Bounds the change in GCI-hat under a
    delta-perturbation of the alternative-model set."""
    if n_units == 0:
        return 0.0
    k = sum(1 for d in min_distances.values() if theta - delta <= d <= theta + delta)
    return k / n_units


# --------------------------------------------------------------------------
# Convenience: full report
# --------------------------------------------------------------------------

def report(
    partition: Dict[tuple, Prov],
    events: Sequence[Event],
    pattern_set: Set[str],
    reconstructed: Set[str],
    beta: Sequence[float] = (0.2, 0.2, 0.2, 0.2, 0.2),
) -> Dict[str, float]:
    from .model import profile as _profile
    plc, prf, ppa_v = ppa(events, reconstructed)
    d, l, g = drr(partition), lrd_true(partition), gci_true(partition)
    c = ppc(events, pattern_set)
    v = component_vector(d, l, c, ppa_v, g)
    loss, ceil = information_loss(_profile(partition))
    return {
        "DRR": d, "LRD": l, "PPC": c, "PPC_U": ppc_units(events, pattern_set),
        "PLC": plc, "PRF": prf, "PPA": ppa_v, "GCI": g,
        "HRS": hrs(v, beta), "info_loss_bits": loss, "info_loss_ceiling": ceil,
        **{f"n_{k.value}": n for k, n in counts(partition).items()},
    }
