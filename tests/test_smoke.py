"""Smoke test — package imports without error."""
import phoenix2pytest


def test_import():
    assert phoenix2pytest.__version__ == "0.0.1"
