"""Smoke test — verifies the package imports and tooling runs end to end."""

import swarm_field


def test_package_has_version() -> None:
    assert swarm_field.__version__ == "0.1.0"
