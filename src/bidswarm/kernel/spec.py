"""PolicySpec: a policy as pure data (template name + numeric parameters).

Part of the portable kernel (Python 3.9, numpy + stdlib only). A spec is
validated against the registry before use: the name must be registered and
every parameter must be a finite number inside the template's box bounds.
Specs never carry code; nothing here calls ``eval`` or imports by name.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

from .policies import BasePolicy
from .registry import REGISTRY, get_template, registered_names


class PolicySpecError(ValueError):
    """Raised when a spec is malformed, unknown or out of bounds."""


@dataclass(frozen=True)
class PolicySpec:
    name: str
    params: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "params": dict((k, float(self.params[k])) for k in sorted(self.params)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PolicySpec":
        if not isinstance(data, Mapping):
            raise PolicySpecError("spec must be a mapping, got %s" % type(data).__name__)
        extra = set(data) - {"name", "params"}
        if extra:
            raise PolicySpecError("unknown spec fields: %s" % sorted(extra))
        if "name" not in data or "params" not in data:
            raise PolicySpecError("spec needs 'name' and 'params'")
        if not isinstance(data["params"], Mapping):
            raise PolicySpecError("spec params must be a mapping")
        return cls(name=str(data["name"]), params=dict(data["params"]))


def canonical_json(obj: Any) -> str:
    """Deterministic JSON text: sorted keys, no whitespace, no NaN/Infinity."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def spec_hash(spec: PolicySpec) -> str:
    """sha256 hex digest of the spec's canonical JSON."""
    return hashlib.sha256(canonical_json(spec.to_dict()).encode("utf-8")).hexdigest()


def validate_spec(spec: PolicySpec) -> PolicySpec:
    """Check a spec and return a copy whose params are plain floats."""
    if spec.name not in REGISTRY:
        raise PolicySpecError(
            "unknown policy template %r; registered: %s" % (spec.name, registered_names())
        )
    template = get_template(spec.name)
    expected = template.param_names()
    missing = sorted(set(expected) - set(spec.params))
    unknown = sorted(set(spec.params) - set(expected))
    if missing or unknown:
        raise PolicySpecError(
            "%s params mismatch: missing=%s unknown=%s" % (spec.name, missing, unknown)
        )
    clean = {}
    for p in template.PARAM_DEFS:
        value = spec.params[p.name]
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise PolicySpecError("%s.%s must be a real number, got %r" % (spec.name, p.name, value))
        value = float(value)
        if not math.isfinite(value):
            raise PolicySpecError("%s.%s must be finite, got %r" % (spec.name, p.name, value))
        if value < p.low or value > p.high:
            raise PolicySpecError(
                "%s.%s=%r outside bounds [%r, %r]" % (spec.name, p.name, value, p.low, p.high)
            )
        clean[p.name] = value
    problem = template.check_constraints(clean)
    if problem:
        raise PolicySpecError("%s: %s" % (spec.name, problem))
    return PolicySpec(name=spec.name, params=clean)


def default_spec(name: str) -> PolicySpec:
    """The template's untuned default parameters as a validated spec."""
    if name not in REGISTRY:
        raise PolicySpecError("unknown policy template %r" % name)
    return validate_spec(PolicySpec(name=name, params=get_template(name).default_params()))


def spec_from_normalized(name: str, u: Sequence[float]) -> PolicySpec:
    """Build a validated spec from a point of the template's unit cube."""
    if name not in REGISTRY:
        raise PolicySpecError("unknown policy template %r" % name)
    return validate_spec(PolicySpec(name=name, params=get_template(name).from_normalized(u)))


def spec_to_normalized(spec: PolicySpec) -> list:
    spec = validate_spec(spec)
    return get_template(spec.name).to_normalized(spec.params)


def make_policy(
    spec: PolicySpec,
    budget: float,
    cpa_target: float,
    category: Any = None,
    num_time_steps: int = 48,
) -> BasePolicy:
    """Validate ``spec`` and instantiate its policy for one advertiser."""
    spec = validate_spec(spec)
    cls = get_template(spec.name)
    return cls(budget, cpa_target, category, num_time_steps, **spec.params)
