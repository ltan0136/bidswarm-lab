"""Portable policy kernel.

Everything under ``bidswarm.kernel`` must run on Python 3.9 with only numpy and
the standard library, and must not import other ``bidswarm`` subpackages: the
AuctionNet shim will load these files inside the upstream (Python 3.9.12)
process. ``tests/test_kernel_compat.py`` enforces these rules.
"""

from __future__ import annotations

from .policies import (
    BasePolicy,
    DualControlPolicy,
    FixedRatePolicy,
    ParamDef,
    PolicyError,
    PolicyInputError,
    PolicyOutputError,
)
from .registry import REGISTRY, get_template, registered_names
from .spec import (
    PolicySpec,
    PolicySpecError,
    canonical_json,
    default_spec,
    make_policy,
    spec_from_normalized,
    spec_hash,
    spec_to_normalized,
    validate_spec,
)

__all__ = [
    "BasePolicy",
    "DualControlPolicy",
    "FixedRatePolicy",
    "ParamDef",
    "PolicyError",
    "PolicyInputError",
    "PolicyOutputError",
    "PolicySpec",
    "PolicySpecError",
    "REGISTRY",
    "canonical_json",
    "default_spec",
    "get_template",
    "make_policy",
    "registered_names",
    "spec_from_normalized",
    "spec_hash",
    "spec_to_normalized",
    "validate_spec",
]
