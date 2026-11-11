"""Tests: the worked example (Section VII), the basic properties (Prop. 1) and
the theorems (Thm 3 one-sidedness, Thm 4 information loss, Thm 6 calibration,
Thm 7 Lipschitz)."""
import math

import numpy as np
import pytest

from reuse_metrics import (
    Attribution, Event, Prov, ProtocolParams, System, Unit, calibrate_weights,
    component_vector, counts, drr, generate_system, gci_true, hrs, hrs_lipschitz_bound,
    hrs_separated, information_loss, lrd_true, lrd_typicality_expectation, partition,
    ppa, ppc, ppc_units, profile, report,
)
from reuse_metrics.partition import ai_generated_units


# ---------------------------------------------------------------- worked example

def worked_example():
    """Table II: 30 D, 30 P, 80 L, 40 G, 20 H over 200 lines, 5 events."""
    units, attrs, truth = [], {}, {}
    spec = [(Prov.D, 30), (Prov.P, 30), (Prov.L, 80), (Prov.G, 40), (Prov.H, 20)]
    i = 0
    for c, n in spec:
        for _ in range(n):
            u = Unit("bookmarks.py", i, "r1", f"line {i}")
            units.append(u); truth[u.key] = c
            if c == Prov.D:
                attrs[u.key] = Attribution("flask", "import" if i >= 20 else "copy")
            i += 1
    keys = {c: [u.key for u in units if truth[u.key] == c] for c in Prov}
    events = [
        Event("e1", 1, "a SQLAlchemy model for bookmarks", "m", frozenset(keys[Prov.P])),
        Event("e2", 2, "add CRUD routes", "m", frozenset(keys[Prov.L][:40])),
        Event("e3", 3, "add CRUD routes for tags", "m", frozenset(keys[Prov.L][40:])),
        Event("e4", 4, "write a tag-cloud algorithm", "m", frozenset(keys[Prov.G])),
        Event("e5", 5, "tweak the response format", "m", frozenset()),
    ]
    return System(units, attrs, events), truth


def test_worked_example_numbers():
    system, truth = worked_example()
    N = {"SQLAlchemy"}
    r = report(truth, system.events, N, reconstructed={"e1", "e2", "e3", "e4"},
               beta=(0.30, 0.20, 0.15, 0.15, 0.20))
    assert r["DRR"] == pytest.approx(0.15)
    assert r["LRD"] == pytest.approx(0.40)
    assert r["PPC"] == pytest.approx(0.20)
    assert r["PPC_U"] == pytest.approx(30 / 150)
    assert r["PPA"] == pytest.approx(0.80)
    assert r["GCI"] == pytest.approx(0.20)
    assert r["HRS"] == pytest.approx(0.475)
    assert r["info_loss_bits"] == pytest.approx(1.5366, abs=1e-3)
    assert r["info_loss_ceiling"] == pytest.approx(1.70)


def test_lrd_typicality_expectation_matches_paper():
    p = {Prov.D: .15, Prov.L: .40, Prov.P: .15, Prov.G: .20, Prov.H: .10}
    e = lrd_typicality_expectation(p, {Prov.L: .9, Prov.P: .9, Prov.G: .9}, fp_human=.2)
    assert e == pytest.approx(0.695)


def test_sensitivity_table_hrs_range():
    beta = (0.30, 0.20, 0.15, 0.15, 0.20)
    lo = hrs(component_vector(0.14, 0.48, 1 / 6, 0.60, 0.23), beta)
    hi = hrs(component_vector(0.16, 0.32, 0.25, 1.00, 0.18), beta)
    assert 0.41 <= lo and hi <= 0.54


# ---------------------------------------------------------------- Proposition 1

def test_bounded_and_separated_recovers_hrs():
    v = component_vector(0.15, 0.40, 0.20, 0.80, 0.20)
    beta = np.array([0.30, 0.20, 0.15, 0.15, 0.20])
    assert 0 <= hrs(v, beta) <= 1
    gamma = beta[[0, 1, 4]].sum()
    sep = hrs_separated(v, beta[[0, 1, 4]] / gamma, beta[[2, 3]] / (1 - gamma), gamma)
    assert sep == pytest.approx(hrs(v, beta))
    assert v.min() <= hrs(v, beta) <= v.max()


def test_drr_monotone_under_deliberate_addition():
    system, truth = worked_example()
    before = drr(truth)
    truth2 = dict(truth); truth2[("x.py", 0, "r1")] = Prov.D
    assert drr(truth2) - before == pytest.approx((200 - 30) / (200 * 201))


# ---------------------------------------------------------------- Theorem 3

@pytest.mark.parametrize("seed", range(5))
def test_gci_estimate_is_one_sided_and_monotone_in_alt_models(seed):
    system, truth, oracle, N = generate_system(seed=seed, models=("A", "B", "C"))
    base = dict(rho0=0.5, pattern_set=N, theta=0.4)
    small = partition(system, ProtocolParams(alt_models={"A", "B"}, **base), oracle)
    large = partition(system, ProtocolParams(alt_models={"A", "B", "C"}, **base), oracle)
    assert gci_true(small) >= gci_true(large)          # Thm 3(ii)
    assert gci_true(large) >= gci_true(truth)          # Thm 3(i): upper estimate
    # exactness: with every model and an exact oracle the estimate equals the truth
    assert gci_true(large) == pytest.approx(gci_true(truth))


def test_swap_passers_are_never_G():
    system, truth, oracle, N = generate_system(seed=1)
    est = partition(system, ProtocolParams(alt_models={"model-A", "model-B"}, pattern_set=N, theta=0.4), oracle)
    for k, c in est.items():
        if truth[k] == Prov.G and c != Prov.H:   # edited-away units become H
            assert c == Prov.G


def test_edit_threshold_moves_units_to_H():
    system, truth, oracle, N = generate_system(seed=2, edit_fraction=0.3)
    ai = ai_generated_units(system, rho0=0.5)
    for k, c in truth.items():
        assert (k in ai) == (c in {Prov.L, Prov.P, Prov.G})


# ---------------------------------------------------------------- Theorem 4

def test_information_loss_zero_when_non_D_mass_in_one_class():
    loss, ceil = information_loss({Prov.D: .25, Prov.L: .75, Prov.P: 0, Prov.G: 0, Prov.H: 0})
    assert loss == 0 and ceil == pytest.approx(1.5)


def test_information_loss_max_at_uniform():
    loss, ceil = information_loss({Prov.D: .2, Prov.L: .2, Prov.P: .2, Prov.G: .2, Prov.H: .2})
    assert loss == pytest.approx(ceil)


# ---------------------------------------------------------------- Theorem 6

def test_calibration_uniqueness_condition():
    rng = np.random.default_rng(0)
    X = rng.random((8, 5))
    h = X @ np.array([.3, .2, .15, .15, .2])
    beta, unique = calibrate_weights(X, h)
    assert unique and np.allclose(beta, [.3, .2, .15, .15, .2], atol=1e-3)
    # the paper's counterexample: rows 1, e1, e2, e3 -> rank 4, A != {0}
    X2 = np.array([[1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]], float)
    _, unique2 = calibrate_weights(X2, X2 @ np.full(5, .2))
    assert not unique2
    # one column twice another, others independent -> still unique
    X3 = rng.random((8, 5)); X3[:, 0] = 2 * X3[:, 1]
    _, unique3 = calibrate_weights(X3, X3 @ np.full(5, .2))
    assert unique3


# ---------------------------------------------------------------- Theorem 7

@pytest.mark.parametrize("seed", range(20))
def test_lipschitz_bound_holds(seed):
    rng = np.random.default_rng(seed)
    v = rng.random(5)
    b1 = rng.dirichlet(np.ones(5)); b2 = rng.dirichlet(np.ones(5))
    assert abs(hrs(v, b1) - hrs(v, b2)) <= hrs_lipschitz_bound(v, b1, b2) + 1e-12


def test_lipschitz_bound_is_tight():
    v = np.array([0.1, 0.5, 0.9, 0.3, 0.6])
    b1 = np.zeros(5); b1[np.argmax(v)] = 1
    b2 = np.zeros(5); b2[np.argmin(v)] = 1
    assert abs(hrs(v, b1) - hrs(v, b2)) == pytest.approx(hrs_lipschitz_bound(v, b1, b2))


# ---------------------------------------------------------------- model guards

def test_single_origin_enforced():
    u = Unit("a.py", 0, "r1", "x")
    with pytest.raises(ValueError):
        System([u], {}, [Event("e1", 1, "p", "m", frozenset([u.key])),
                         Event("e2", 2, "p", "m", frozenset([u.key]))])
