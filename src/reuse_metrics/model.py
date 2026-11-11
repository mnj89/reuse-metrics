"""Data model: located code units, generation events, provenance classes.

Mirrors Section IV of the paper. Units are *located* (file, position, revision),
so two textually identical lines are distinct elements and may carry different
provenance classes (Definition 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, Iterable, List, Optional


class Prov(str, Enum):
    """The five provenance classes (Definition 2)."""

    D = "D"  # deliberate reuse (attributed import / call / copy / fork)
    L = "L"  # latent reuse
    P = "P"  # prompted reuse
    G = "G"  # generative coupling
    H = "H"  # human-authored, no AI mediation


@dataclass(frozen=True)
class Unit:
    """A located code unit u = (file, position, revision) with content c(u)."""

    file: str
    pos: int
    rev: str
    content: str = ""

    @property
    def key(self) -> tuple:
        return (self.file, self.pos, self.rev)


@dataclass(frozen=True)
class Attribution:
    """An attribution record r(u) for a D-class unit."""

    source: str
    kind: str  # import | call | copy | fork

    def __post_init__(self) -> None:
        if self.kind not in {"import", "call", "copy", "fork"}:
            raise ValueError(f"unknown attribution kind: {self.kind}")


@dataclass
class Event:
    """A generation event e = (t, p, m, U_e); U_e is a set of unit keys."""

    event_id: str
    timestamp: float
    prompt: str
    model: str
    units: FrozenSet[tuple]
    prompt_stored: bool = True
    # Per unit: emitted text at generation time (for the edit-distance rule).
    emitted_text: Dict[tuple, str] = field(default_factory=dict)
    # Function/method name per unit for regeneration blocks; None = module level.
    block_of: Dict[tuple, Optional[str]] = field(default_factory=dict)


@dataclass
class System:
    """A software system: units, attribution records and the event log E."""

    units: List[Unit]
    attributions: Dict[tuple, Attribution] = field(default_factory=dict)
    events: List[Event] = field(default_factory=list)

    def __post_init__(self) -> None:
        keys = [u.key for u in self.units]
        if len(keys) != len(set(keys)):
            raise ValueError("units must be distinct located triples")
        seen: Dict[tuple, str] = {}
        for e in self.events:
            for k in e.units:
                if k in seen:
                    raise ValueError(
                        f"unit {k} emitted by two events ({seen[k]}, {e.event_id}); "
                        "Assumption 1 requires a single origin"
                    )
                seen[k] = e.event_id

    @property
    def size(self) -> int:
        return len(self.units)

    def event_of(self) -> Dict[tuple, Event]:
        return {k: e for e in self.events for k in e.units}


def regeneration_block(event: Event, key: tuple) -> tuple:
    """Block id b(u): units of one event in one function, or the event's
    module-level units as one further block."""
    return (event.event_id, event.block_of.get(key))


def counts(partition: Dict[tuple, Prov]) -> Dict[Prov, int]:
    out = {c: 0 for c in Prov}
    for c in partition.values():
        out[c] += 1
    return out


def profile(partition: Dict[tuple, Prov]) -> Dict[Prov, float]:
    """Normalised provenance profile p(U)."""
    n = len(partition)
    return {c: (k / n if n else 0.0) for c, k in counts(partition).items()}
