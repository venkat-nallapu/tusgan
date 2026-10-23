"""Shared pytest fixtures for the TUS-GAN test suite.

v4/generator.py and v4/critic.py have no __init__.py, so this file puts the
repo root on sys.path and lets pytest import them as implicit namespace
packages (`from v4.generator import Generator`).
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@pytest.fixture(autouse=True)
def seed_everything():
    torch.manual_seed(0)


@pytest.fixture
def batch_size():
    return 4
