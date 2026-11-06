"""Environment backends and the backend factory.

Backends are selected explicitly by name. An unknown name is a configuration
error; a backend whose setup is missing (for AuctionNet: the pinned checkout
or interpreter) raises; there is never a silent fallback to ``toy``.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..types import BACKEND_AUCTIONNET, BACKEND_TOY, ConfigError
from . import auctionnet, toy
from .auctionnet import AuctionNetRunnerError, AuctionNetSetupError
from .base import BackendNotIntegratedError, Environment, purpose_for

KNOWN_BACKENDS = (BACKEND_TOY, BACKEND_AUCTIONNET)

INTEGRATION_STATUS = {
    BACKEND_TOY: toy.INTEGRATION_STATUS,
    BACKEND_AUCTIONNET: auctionnet.INTEGRATION_STATUS,
}


def _backend_name(cfg: Mapping[str, Any]) -> str:
    backend = cfg.get("backend")
    if backend not in KNOWN_BACKENDS:
        raise ConfigError(f"[environment] unknown backend {backend!r}; known: {list(KNOWN_BACKENDS)}")
    return backend


def validate_environment_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return the validated environment config with defaults filled in."""
    backend = _backend_name(cfg)
    if backend == BACKEND_TOY:
        return toy.ToyConfig.from_mapping(cfg).to_dict()
    return auctionnet.validate_config(cfg)


def make_environment(cfg: Mapping[str, Any]) -> Environment:
    backend = _backend_name(cfg)
    if backend == BACKEND_TOY:
        return toy.ToyEnvironment(toy.ToyConfig.from_mapping(cfg))
    return auctionnet.AuctionNetEnvironment(cfg)


__all__ = [
    "AuctionNetRunnerError",
    "AuctionNetSetupError",
    "INTEGRATION_STATUS",
    "KNOWN_BACKENDS",
    "BackendNotIntegratedError",
    "Environment",
    "make_environment",
    "purpose_for",
    "validate_environment_config",
]
