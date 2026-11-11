"""Synthetic system and prompt-log generator for testing the protocol.

The generator plants a *known* true partition, which lets tests compare the
operational estimator against ground truth, as the laboratory study (Study B)
will do with real logs.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .model import Attribution, Event, Prov, System, Unit

_PATTERNS = ["SQLAlchemy model", "Flask blueprint", "JWT auth", "repository pattern", "pydantic schema"]
_GENERIC = ["add CRUD routes", "write a helper", "parse the config", "implement the handler", "add logging"]


@dataclass
class SyntheticOracle:
    """Regeneration oracle: returns d^{(q)}(b; model). Blocks planted as G are
    regenerable only by their originating model; under-specified blocks are
    regenerable by nobody; the rest by every model.

    Distances are drawn around `near` (regenerable) or `far` (not); the gap
    between them controls how many blocks sit close to theta (Theorem 5)."""

    truth: Dict[tuple, str]            # block -> "G" | "regen" | "under"
    origin: Dict[tuple, str]           # block -> originating model
    near: float = 0.1
    far: float = 0.8
    noise: float = 0.05
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._cache: Dict[Tuple[tuple, str], float] = {}

    def __call__(self, block: tuple, model: str, prompt: str) -> float:
        key = (block, model)
        if key not in self._cache:
            kind = self.truth.get(block, "regen")
            if kind == "under":
                centre = self.far
            elif kind == "G":
                centre = self.near if model == self.origin.get(block) else self.far
            else:
                centre = self.near
            self._cache[key] = min(1.0, max(0.0, self._rng.gauss(centre, self.noise)))
        return self._cache[key]


def generate_system(
    n_units: int = 200,
    weights: Dict[Prov, float] = None,
    n_events: int = 8,
    models: Tuple[str, ...] = ("model-A", "model-B"),
    edit_fraction: float = 0.1,
    under_specified_fraction: float = 0.1,
    seed: int = 0,
) -> Tuple[System, Dict[tuple, Prov], SyntheticOracle, Set[str]]:
    """Returns (system, true_partition, oracle, pattern_set).

    `weights` gives the target class proportions. A fraction `edit_fraction`
    of AI units are heavily edited (edit distance > rho0), which moves them to
    H under Assumption 1 and is reflected in `true_partition`."""
    rng = random.Random(seed)
    weights = weights or {Prov.D: .15, Prov.L: .40, Prov.P: .15, Prov.G: .20, Prov.H: .10}
    classes = list(weights)
    truth: Dict[tuple, Prov] = {}
    units: List[Unit] = []
    attributions: Dict[tuple, Attribution] = {}
    ai_keys: List[tuple] = []
    for i in range(n_units):
        c = rng.choices(classes, weights=[weights[k] for k in classes])[0]
        u = Unit("app.py", i, "r1", content=f"line {i} {c.value} " + " ".join(rng.choices("abcdefgh", k=6)))
        units.append(u)
        truth[u.key] = c
        if c == Prov.D:
            attributions[u.key] = Attribution("some-lib", rng.choice(["import", "copy"]))
        elif c != Prov.H:
            ai_keys.append(u.key)

    # Group AI units into events by class, then into function blocks.
    events: List[Event] = []
    oracle_truth: Dict[tuple, str] = {}
    oracle_origin: Dict[tuple, str] = {}
    by_class: Dict[Prov, List[tuple]] = {Prov.L: [], Prov.P: [], Prov.G: []}
    for k in ai_keys:
        by_class[truth[k]].append(k)
    eid = 0
    content_by_key = {u.key: u.content for u in units}
    for c, keys in by_class.items():
        rng.shuffle(keys)
        n_ev = max(1, round(n_events * len(keys) / max(1, len(ai_keys))))
        chunks = [keys[j::n_ev] for j in range(n_ev)] if keys else []
        for chunk in chunks:
            if not chunk:
                continue
            model = rng.choice(models)
            prompt = rng.choice(_PATTERNS) if c == Prov.P else rng.choice(_GENERIC)
            if c == Prov.G:
                prompt = "bespoke " + rng.choice(_GENERIC)
            emitted = {k: content_by_key[k] for k in chunk}
            # Heavily edited units: emitted text differs completely -> become H.
            for k in chunk:
                if rng.random() < edit_fraction:
                    emitted[k] = "totally different emitted text here"
                    truth[k] = Prov.H
            block_of = {k: f"fn_{eid}_{j % 2}" for j, k in enumerate(chunk)}
            ev = Event(f"e{eid}", float(eid), prompt, model, frozenset(chunk),
                       prompt_stored=True, emitted_text=emitted, block_of=block_of)
            events.append(ev)
            for b in set((ev.event_id, v) for v in block_of.values()):
                kind = "G" if c == Prov.G else "regen"
                if c != Prov.G and rng.random() < under_specified_fraction:
                    kind = "under"
                oracle_truth[b] = kind
                oracle_origin[b] = model
            eid += 1
    system = System(units, attributions, events)
    oracle = SyntheticOracle(oracle_truth, oracle_origin, seed=seed)
    return system, truth, oracle, set(_PATTERNS)
