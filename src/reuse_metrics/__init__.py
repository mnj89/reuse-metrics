"""reuse-metrics: reference implementation of the provenance-partition
framework for measuring software reuse in AI-augmented codebases."""

from .model import Attribution, Event, Prov, System, Unit, counts, profile
from .partition import ProtocolParams, partition, swap_test, ai_generated_units
from .metrics import (
    calibrate_weights, component_vector, drr, drr_size, gci_estimate, gci_robustness_F,
    gci_true, hrs, hrs_lipschitz_bound, hrs_separated, information_loss,
    lrd_true, lrd_typicality, lrd_typicality_expectation, ppa, ppc, ppc_units, report,
)
from .synthetic import SyntheticOracle, generate_system

__version__ = "0.1.0"
__all__ = [n for n in dir() if not n.startswith("_")]
