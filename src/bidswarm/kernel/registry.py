"""Name -> policy template registry.

Part of the portable kernel (Python 3.9, numpy + stdlib only). The registry is
a closed, explicit mapping: specs can only refer to templates listed here, and
nothing is ever imported or evaluated from a spec.
"""

from __future__ import annotations

from typing import Dict, List, Type

from .policies import BasePolicy, DualControlPolicy, FixedRatePolicy

REGISTRY: Dict[str, Type[BasePolicy]] = dict(
    (cls.name, cls) for cls in (FixedRatePolicy, DualControlPolicy)
)


def registered_names() -> List[str]:
    return sorted(REGISTRY)


def get_template(name: str) -> Type[BasePolicy]:
    """Return the policy class registered under ``name`` (KeyError if absent)."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(
            "unknown policy template %r; registered: %s" % (name, registered_names())
        ) from None
