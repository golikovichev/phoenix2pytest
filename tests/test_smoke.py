"""Smoke test: package imports without error."""

import re

import phoenix2pytest


def test_import():
    assert re.fullmatch(r"\d+\.\d+\.\d+", phoenix2pytest.__version__)
