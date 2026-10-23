"""Unit tests for Section 6.8 sensitivity sweep modules."""

import os
import sys
import tempfile
import pytest
import torch
import numpy as np

# Ensure root and sensitivity sweep directories are in path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SWEEP_DIR = os.path.join(ROOT_DIR, "experiments/sensitivity_sweep")
if SWEEP_DIR not in sys.path:
    sys.path.insert(0, SWEEP_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from train_sweep import compute_neuro_symbolic_loss
from make_figure10_TRC import get_sensitivity_data, render_figure10


class TestSensitivityLoss:
    def test_child_employment_loss_penalty(self):
        # 4 samples, 9 channels, 48 slots
        probs = torch.zeros(4, 9, 48)
        cond = torch.zeros(4, 82)

        # Sample 0 & 1 are children (cond[0]=1), sample 2 & 3 are adults (cond[0]=0)
        cond[0, 0] = 1.0
        cond[1, 0] = 1.0

        # Sample 0 has 100% employment (channel 0)
        probs[0, 0, :] = 1.0
        # Sample 1 has 50% employment
        probs[1, 0, :] = 0.5
        # Sample 2 (adult) has 100% employment
        probs[2, 0, :] = 1.0

        loss = compute_neuro_symbolic_loss(probs, cond)
        # Expected loss is mean over children: (1.0 + 0.5) / 2 = 0.75
        assert torch.isclose(loss, torch.tensor(0.75), atol=1e-5)

    def test_adult_only_batch_yields_zero_loss(self):
        probs = torch.ones(4, 9, 48)
        cond = torch.zeros(4, 82)  # No children
        loss = compute_neuro_symbolic_loss(probs, cond)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-7)


class TestFigure10Renderer:
    def test_get_sensitivity_data_returns_valid_structure(self):
        records = get_sensitivity_data()
        assert len(records) >= 4
        for r in records:
            assert "lambda_rule" in r
            assert "raw_violation_rate_pct" in r
            assert "macro_jsd" in r

    def test_render_figure10_generates_pdf_and_png(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = get_sensitivity_data()
            render_figure10(output_dir=tmpdir, records=records, no_mirror=True)

            pdf_file = os.path.join(tmpdir, "figure10_sensitivity_tradeoff.pdf")
            png_file = os.path.join(tmpdir, "figure10_sensitivity_tradeoff.png")

            assert os.path.exists(pdf_file)
            assert os.path.exists(png_file)
            assert os.path.getsize(pdf_file) > 10000  # Non-trivial PDF
            assert os.path.getsize(png_file) > 50000  # Non-trivial PNG

    def test_render_figure10_polymorphic_args(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = get_sensitivity_data()
            # Test (records, output_dir) argument ordering
            render_figure10(records, tmpdir, no_mirror=True)
            assert os.path.exists(os.path.join(tmpdir, "figure10_sensitivity_tradeoff.png"))
