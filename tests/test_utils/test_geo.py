"""Tests for wavy.utils.geo."""

import pytest
import numpy as np

from wavy.utils.geo import (
    convert_meteorologic_oceanographic,
    footprint_pulse_limited_radius,
    haversine_np,
    haversineA,
    haversineP,
)


class TestHaversine:
    # Oslo (10.74E, 59.91N) → London (0.13W, 51.51N) ≈ 1156 km
    lon1, lat1 = 10.74, 59.91
    lon2, lat2 = -0.13, 51.51
    expected_km = 1156

    def test_haversineP_approx(self):
        d = haversineP(self.lon1, self.lat1, self.lon2, self.lat2)
        assert abs(d - self.expected_km) < 20

    def test_haversine_np_approx(self):
        d = haversine_np(self.lon1, self.lat1, self.lon2, self.lat2)
        assert abs(d - self.expected_km) < 20

    def test_haversineA_scalar_returns_list(self):
        d = haversineA(self.lon1, self.lat1, self.lon2, self.lat2)
        assert isinstance(d, list)
        assert abs(d[0] - self.expected_km) < 20

    def test_haversineA_list(self):
        lons1 = [self.lon1, self.lon1]
        lats1 = [self.lat1, self.lat1]
        lons2 = [self.lon2, self.lon2]
        lats2 = [self.lat2, self.lat2]
        d = haversineA(lons1, lats1, lons2, lats2)
        assert len(d) == 2
        assert all(abs(v - self.expected_km) < 20 for v in d)

    def test_same_point_zero_distance(self):
        assert haversineP(10.0, 60.0, 10.0, 60.0) == pytest.approx(0.0, abs=1e-9)

    def test_scalar_and_np_agree(self):
        d_p = haversineP(self.lon1, self.lat1, self.lon2, self.lat2)
        d_np = haversine_np(self.lon1, self.lat1, self.lon2, self.lat2)
        assert abs(d_p - d_np) < 1e-6


class TestConvertMeteorologicOceanographic:
    def test_180_shift(self):
        assert convert_meteorologic_oceanographic(0) == 180
        assert convert_meteorologic_oceanographic(180) == 0

    def test_wraps_modulo_360(self):
        assert convert_meteorologic_oceanographic(270) == 90

    def test_double_conversion_is_identity(self):
        for alpha in [0, 45, 90, 135, 180, 270, 359]:
            assert convert_meteorologic_oceanographic(
                convert_meteorologic_oceanographic(alpha)
            ) == alpha


class TestFootprintPulseLimitedRadius:
    # Typical Envisat/Jason values: h=800 km, tau=3.125 ns
    h = 800e3
    tau = 3.125e-9

    def test_returns_positive(self):
        r = footprint_pulse_limited_radius(Hs=2.0, h=self.h, tau=self.tau)
        assert r > 0

    def test_increases_with_Hs(self):
        r1 = footprint_pulse_limited_radius(Hs=1.0, h=self.h, tau=self.tau)
        r2 = footprint_pulse_limited_radius(Hs=4.0, h=self.h, tau=self.tau)
        assert r2 > r1

    def test_increases_with_altitude(self):
        r1 = footprint_pulse_limited_radius(Hs=2.0, h=500e3, tau=self.tau)
        r2 = footprint_pulse_limited_radius(Hs=2.0, h=1000e3, tau=self.tau)
        assert r2 > r1
