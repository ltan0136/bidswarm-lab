"""CLI doctor output."""

from __future__ import annotations

from bidswarm.cli import main


def test_doctor_reports_versions_kernel_check_and_backend_status(capsys):
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "python:" in out and "numpy:" in out
    assert "kernel Python 3.9 syntax check: ok" in out
    assert "backend auctionnet: integration_status=not_integrated" in out
    assert "does not install anything" in out
