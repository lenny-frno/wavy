"""Tests for wavy.utils.stats."""

import numpy as np
import pytest

from wavy.utils.stats import (
    bootstr,
    calc_deep_water_T,
    compute_quantiles,
    marginalize,
    runmean,
    wave_length_mask_swim,
)


class TestRunmean:
    vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    def test_centered_mean(self):
        out, _ = runmean(self.vec, win=3, mode="centered")
        assert out[3] == pytest.approx(4.0)

    def test_left_mean(self):
        out, _ = runmean(self.vec, win=3, mode="left")
        assert out[2] == pytest.approx(2.0)

    def test_right_mean(self):
        out, _ = runmean(self.vec, win=3, mode="right")
        assert out[0] == pytest.approx(2.0)

    def test_centered_nan_at_boundary_edges(self):
        out, _ = runmean(self.vec, win=3, mode="centered")
        assert np.isnan(out[0])
        assert np.isnan(out[-1])

    def test_output_same_length_as_input(self):
        out, std = runmean(self.vec, win=3, mode="centered")
        assert len(out) == len(self.vec)
        assert len(std) == len(self.vec)

    def test_uniform_signal_returns_self(self):
        uniform = np.ones(10)
        out, std = runmean(uniform, win=3, mode="left")
        # Non-NaN values should all equal 1.0
        valid = out[~np.isnan(out)]
        assert np.allclose(valid, 1.0)


class TestMarginalize:
    def test_removes_nans_single_array(self):
        a = np.array([1.0, np.nan, 3.0, np.nan])
        result = marginalize(a)
        assert len(result) == 2
        assert not np.any(np.isnan(result))

    def test_removes_nans_paired_arrays(self):
        a = np.array([1.0, np.nan, 3.0])
        b = np.array([4.0, 5.0, np.nan])
        a1, b1, idx = marginalize(a, b)
        assert len(a1) == 1
        assert a1[0] == pytest.approx(1.0)
        assert b1[0] == pytest.approx(4.0)

    def test_no_nans_returns_full_array(self):
        a = np.array([1.0, 2.0, 3.0])
        result = marginalize(a)
        assert len(result) == 3


class TestBootstr:
    def test_output_shape(self):
        a = np.arange(10.0)
        b, bidx = bootstr(a, reps=5)
        assert b.shape == (10, 5)
        assert bidx.shape == (10, 5)

    def test_indices_in_valid_range(self):
        a = np.arange(8.0)
        _, bidx = bootstr(a, reps=3)
        assert bidx.min() >= 0
        assert bidx.max() < len(a)

    def test_drawn_values_within_input(self):
        a = np.array([10.0, 20.0, 30.0])
        b, _ = bootstr(a, reps=100)
        assert set(b.flatten()).issubset({10.0, 20.0, 30.0})


class TestComputeQuantiles:
    def test_median(self):
        ts = np.arange(1.0, 11.0)
        q = compute_quantiles(ts, [0.5])
        assert q[0] == pytest.approx(5.5)

    def test_min_max(self):
        ts = np.arange(1.0, 11.0)
        q = compute_quantiles(ts, [0.0, 1.0])
        assert q[0] == pytest.approx(1.0)
        assert q[1] == pytest.approx(10.0)

    def test_ignores_nans(self):
        ts = np.array([1.0, np.nan, 3.0])
        q = compute_quantiles(ts, [0.5])
        assert not np.isnan(q[0])

    def test_multiple_quantiles_length(self):
        ts = np.arange(1.0, 101.0)
        lq = [0.25, 0.5, 0.75]
        q = compute_quantiles(ts, lq)
        assert len(q) == 3


class TestCalcDeepWaterT:
    def test_known_wavelength_10s(self):
        # λ = g*T²/(2π) → T = sqrt(λ*2π/g); for λ≈156m, T≈10s
        T = calc_deep_water_T(l=156.0)
        assert abs(T - 10.0) < 0.2

    def test_longer_wavelength_longer_period(self):
        T1 = calc_deep_water_T(l=100.0)
        T2 = calc_deep_water_T(l=400.0)
        assert T2 > T1


class TestWaveLengthMaskSwim:
    # wave_length_mask_swim(ds, llim, ulim) converts wavelengths (m) to
    # deep-water periods via calc_deep_water_T, then keeps only those
    # whose period falls in (llim, ulim) seconds.
    #   λ= 50 m → T≈ 5.7 s
    #   λ=150 m → T≈ 9.8 s
    #   λ=300 m → T≈13.9 s
    #   λ=2500 m → T≈40.0 s
    def test_mask_within_bounds(self):
        import xarray as xr

        lam = xr.DataArray(np.array([50.0, 150.0, 300.0, 2500.0]))
        # keep wavelengths whose period is between 5 s and 30 s
        mask = wave_length_mask_swim(lam, llim=5, ulim=30)
        assert mask.values[0]      # T≈5.7 s → inside (5, 30)
        assert mask.values[1]      # T≈9.8 s → inside (5, 30)
        assert mask.values[2]      # T≈13.9 s → inside (5, 30)
        assert not mask.values[3]  # T≈40 s  → outside (5, 30)
