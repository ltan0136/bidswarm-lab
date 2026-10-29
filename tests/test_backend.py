"""backend = "auctionnet" must fail loudly and never fall back to toy."""

from __future__ import annotations

from pathlib import Path

import pytest

from bidswarm.cli import main
from bidswarm.config import load_config
from bidswarm.environments import BackendNotIntegratedError, make_environment
from bidswarm.environments.auctionnet import INTEGRATION_STATUS, AuctionNetEnvironment
from bidswarm.experiment import run_experiment

from .conftest import CONFIG_DIR


def test_auctionnet_backend_construction_raises():
    assert INTEGRATION_STATUS == "not_integrated"
    with pytest.raises(BackendNotIntegratedError) as info:
        make_environment({"backend": "auctionnet"})
    assert info.value.integration_status == "not_integrated"
    assert "No fallback" in str(info.value)
    assert info.value.blockers
    with pytest.raises(BackendNotIntegratedError):
        AuctionNetEnvironment()


def test_run_with_auctionnet_config_fails_before_writing(tmp_path: Path):
    cfg = load_config(CONFIG_DIR / "auctionnet.example.toml")
    with pytest.raises(BackendNotIntegratedError):
        run_experiment(cfg, out_root=tmp_path / "runs", serial=True)
    assert not (tmp_path / "runs").exists()


def test_cli_run_auctionnet_example_exits_nonzero(tmp_path: Path, capsys):
    out = tmp_path / "runs"
    code = main(["run", "--config", str(CONFIG_DIR / "auctionnet.example.toml"), "--serial", "--out", str(out)])
    assert code == 2
    err = capsys.readouterr().err
    assert "integration_status=not_integrated" in err
    assert not out.exists()
