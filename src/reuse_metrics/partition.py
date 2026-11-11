"""The partitioning protocol (Section VIII-A, tests T1-T4) and the model-swap
regeneration test (Section VIII-C).

The swap test is abstracted behind a `RegenOracle`: a callable that, given a
block, a model name and the prompt, returns the q-quantile distance
d^{(q)}(b; model). The reference implementation ships a synthetic oracle for
testing; a real deployment plugs in LLM calls plus a distance function.
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable, Dict, Iterable, Optional, Set

from .model import Event, Prov, System, regeneration_block

RegenOracle = Callable[[tuple, str, str], float]  # (block, model, prompt) -> d^{(q)}


@dataclass
class ProtocolParams:
    """Parameter tuple that must be reported with every metric value."""

    rho0: float = 0.5           # edit-distance threshold (Assumption 1)
    pattern_set: Set[str] = None  # N, named patterns
    alt_models: Set[str] = None   # M'
    theta: float = 0.3            # swap-test distance threshold
    q: float = 0.5                # regeneration success probability
    R: int = 5                    # samples per model
    temperature: float = 0.2

    def __post_init__(self) -> None:
        self.pattern_set = set(self.pattern_set or ())
        self.alt_models = set(self.alt_models or ())
        if not 0 <= self.rho0 < 1:
            raise ValueError("rho0 must be in [0,1)")
        if not 0 < self.q <= 1:
            raise ValueError("q must be in (0,1]")

    def as_dict(self) -> dict:
        return {
            "rho0": self.rho0, "N": sorted(self.pattern_set),
            "M_alt": sorted(self.alt_models), "theta": self.theta,
            "q": self.q, "R": self.R, "T": self.temperature,
        }


def normalised_edit_distance(a: str, b: str) -> float:
    """1 - similarity ratio on whitespace-split tokens (in [0,1])."""
    ta, tb = a.split(), b.split()
    if not ta and not tb:
        return 0.0
    return 1.0 - SequenceMatcher(None, ta, tb).ratio()


def ai_generated_units(system: System, rho0: float) -> Dict[tuple, Event]:
    """U_AI under Assumption 1: emitted by an event and edited by at most rho0."""
    content = {u.key: u.content for u in system.units}
    out: Dict[tuple, Event] = {}
    for e in system.events:
        for k in e.units:
            if k not in content:
                continue
            emitted = e.emitted_text.get(k)
            if emitted is None or normalised_edit_distance(emitted, content[k]) <= rho0:
                out[k] = e
    return out


def names_pattern(prompt: str, pattern_set: Set[str]) -> bool:
    p = prompt.lower()
    return any(n.lower() in p for n in pattern_set)


def swap_test(
    system: System,
    ai_units: Dict[tuple, Event],
    params: ProtocolParams,
    oracle: Optional[RegenOracle],
) -> Dict[tuple, str]:
    """Per block: 'under_specified', 'coupled' (fails swap), or 'regenerable'.

    Implements Section VIII-C: self-regeneration control under the originating
    model, then the swap over M' \\ {m(u)}; verdict = min_k d_k(b) > theta.
    """
    verdicts: Dict[tuple, str] = {}
    if oracle is None:
        return verdicts
    blocks: Dict[tuple, Event] = {}
    for k, e in ai_units.items():
        blocks.setdefault(regeneration_block(e, k), e)
    for b, e in blocks.items():
        if oracle(b, e.model, e.prompt) > params.theta:
            verdicts[b] = "under_specified"
            continue
        alts = params.alt_models - {e.model}
        if not alts:
            verdicts[b] = "untested"
            continue
        dmin = min(oracle(b, m, e.prompt) for m in alts)
        verdicts[b] = "coupled" if dmin > params.theta else "regenerable"
    return verdicts


def partition(
    system: System,
    params: ProtocolParams,
    oracle: Optional[RegenOracle] = None,
) -> Dict[tuple, Prov]:
    """Operational estimator pi-hat (tests T1 -> T2 -> T3a -> T3b -> T3c -> T4)."""
    ai = ai_generated_units(system, params.rho0)
    verdicts = swap_test(system, ai, params, oracle)
    out: Dict[tuple, Prov] = {}
    for u in system.units:
        k = u.key
        if k in system.attributions:                       # T1
            out[k] = Prov.D
        elif k in ai:                                      # T2
            e = ai[k]
            b = regeneration_block(e, k)
            if verdicts.get(b) == "coupled":               # T3a
                out[k] = Prov.G
            elif names_pattern(e.prompt, params.pattern_set):  # T3b
                out[k] = Prov.P
            else:                                          # T3c
                out[k] = Prov.L
        else:                                              # T4
            out[k] = Prov.H
    return out
