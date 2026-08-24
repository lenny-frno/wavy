import numpy as np
import pytest
from datetime import datetime

from wavy.ice_module import (
    extract_ice_edge_points,
    compute_distance_to_ice_edge,
    get_ice_field,
    get_dist_to_ice_edge,
    EARTH_RADIUS_M,
)


# ---------------------------------------------------------------------#
# helpers
# ---------------------------------------------------------------------#

def make_lonlat_grid(ny, nx, lon0=0.0, lat0=0.0, res=0.1):
    """
    Simple regular lon/lat mesh, spacing `res` degrees, as 2D arrays
    of shape (ny, nx) matching a sic array of the same shape.
    """
    lons_1d = lon0 + np.arange(nx) * res
    lats_1d = lat0 + np.arange(ny) * res
    lons, lats = np.meshgrid(lons_1d, lats_1d)
    return lons, lats


# ---------------------------------------------------------------------#
# extract_ice_edge_points: basic cases
# ---------------------------------------------------------------------#

def test_extract_ice_edge_points_simple_edge():
    # left half ice, right half water -> edge is the column boundary
    sic = np.zeros((10, 10))
    sic[:, :5] = 1.0
    lons, lats = make_lonlat_grid(10, 10)

    edge_lons, edge_lats = extract_ice_edge_points(lons, lats, sic,
                                                    threshold=0.15)

    assert len(edge_lons) == len(edge_lats)
    assert len(edge_lons) > 0
    # all edge points should come from the ice side (column index 4)
    edge_cols = np.round((edge_lons - lons[0, 0]) / 0.1).astype(int)
    assert np.all(edge_cols == 4)


def test_extract_ice_edge_points_no_ice():
    sic = np.zeros((10, 10))
    lons, lats = make_lonlat_grid(10, 10)

    edge_lons, edge_lats = extract_ice_edge_points(lons, lats, sic,
                                                    threshold=0.15)

    assert len(edge_lons) == 0
    assert len(edge_lats) == 0


def test_extract_ice_edge_points_full_ice():
    sic = np.ones((10, 10))
    lons, lats = make_lonlat_grid(10, 10)

    edge_lons, edge_lats = extract_ice_edge_points(lons, lats, sic,
                                                    threshold=0.15)

    assert len(edge_lons) == 0
    assert len(edge_lats) == 0


def test_extract_ice_edge_points_handles_nan_as_water():
    # NaNs (e.g. land) should be treated as non-ice, not crash
    sic = np.zeros((10, 10))
    sic[:, :5] = 1.0
    sic[0, 0] = np.nan
    lons, lats = make_lonlat_grid(10, 10)

    edge_lons, edge_lats = extract_ice_edge_points(lons, lats, sic,
                                                    threshold=0.15)

    assert len(edge_lons) > 0
    assert not np.any(np.isnan(edge_lons))


# ---------------------------------------------------------------------#
# extract_ice_edge_points: polynya (edge case 1)
# ---------------------------------------------------------------------#

def test_polynya_generates_inner_edge_by_default():
    # ice sheet covering the whole grid, with a small enclosed
    # water hole (polynya) in the middle -> without fill_enclosed_water
    # this hole should generate its own edge in addition to the
    # domain-boundary edge
    sic = np.ones((12, 12))
    sic[5:7, 5:7] = 0.0  # 2x2 polynya, does not touch the border
    lons, lats = make_lonlat_grid(12, 12)

    edge_lons, edge_lats = extract_ice_edge_points(
        lons, lats, sic, threshold=0.15,
        fill_enclosed_water=False)

    # edge should include cells adjacent to the polynya, i.e. some
    # edge points located near the hole (rows/cols 4-7)
    cols = np.round((edge_lons - lons[0, 0]) / 0.1).astype(int)
    rows = np.round((edge_lats - lats[0, 0]) / 0.1).astype(int)
    near_hole = np.any((rows >= 4) & (rows <= 7) & (cols >= 4) & (cols <= 7))
    assert near_hole


# def test_fill_enclosed_water_removes_polynya_edge():
#     sic = np.ones((12, 12))
#     sic[5:7, 5:7] = 0.0  # enclosed polynya
#     lons, lats = make_lonlat_grid(12, 12)

#     edge_lons, edge_lats = extract_ice_edge_points(
#         lons, lats, sic, threshold=0.15,
#         fill_enclosed_water=True)

#     # fully ice-covered once the hole is filled -> no edge at all
#     assert len(edge_lons) == 0
#     assert len(edge_lats) == 0


# def test_fill_enclosed_water_does_not_affect_open_ocean():
#     # water touching the domain border must NOT be filled in,
#     # even with fill_enclosed_water=True
#     sic = np.ones((12, 12))
#     sic[:, 0:2] = 0.0  # water strip touching the left border
#     lons, lats = make_lonlat_grid(12, 12)

#     edge_lons, edge_lats = extract_ice_edge_points(
#         lons, lats, sic, threshold=0.15,
#         fill_enclosed_water=True)

#     assert len(edge_lons) > 0
#     cols = np.round((edge_lons - lons[0, 0]) / 0.1).astype(int)
#     assert np.all(cols == 2)  # only the real ice/water boundary remains


# ---------------------------------------------------------------------#
# extract_ice_edge_points: isolated ice floe (edge case 2)
# ---------------------------------------------------------------------#

def test_isolated_floe_generates_edge_by_default():
    # mostly water, with a small isolated ice floe away from
    # the main ice body
    sic = np.zeros((20, 20))
    sic[2:8, 2:8] = 1.0     # main ice body (6x6 = 36 cells)
    sic[15, 15] = 1.0       # isolated single-cell floe
    lons, lats = make_lonlat_grid(20, 20)

    edge_lons, edge_lats = extract_ice_edge_points(
        lons, lats, sic, threshold=0.15,
        min_feature_cells=None)

    cols = np.round((edge_lons - lons[0, 0]) / 0.1).astype(int)
    rows = np.round((edge_lats - lats[0, 0]) / 0.1).astype(int)
    assert np.any((rows == 15) & (cols == 15))


# def test_min_feature_cells_drops_isolated_floe():
#     sic = np.zeros((20, 20))
#     sic[2:8, 2:8] = 1.0     # main ice body, 36 cells
#     sic[15, 15] = 1.0       # isolated single-cell floe

#     lons, lats = make_lonlat_grid(20, 20)

#     edge_lons, edge_lats = extract_ice_edge_points(
#         lons, lats, sic, threshold=0.15,
#         min_feature_cells=5)

#     cols = np.round((edge_lons - lons[0, 0]) / 0.1).astype(int)
#     rows = np.round((edge_lats - lats[0, 0]) / 0.1).astype(int)
#     # the isolated floe cell must no longer appear in the edge set
#     assert not np.any((rows == 15) & (cols == 15))
#     # the main ice body edge must still be present
#     assert len(edge_lons) > 0


# def test_min_feature_cells_keeps_large_features():
#     sic = np.zeros((20, 20))
#     sic[2:8, 2:8] = 1.0  # 36 cells, should survive a low threshold

#     lons, lats = make_lonlat_grid(20, 20)

#     edge_lons, edge_lats = extract_ice_edge_points(
#         lons, lats, sic, threshold=0.15,
#         min_feature_cells=5)

#     assert len(edge_lons) > 0


# def test_min_feature_cells_can_remove_all_ice():
#     # single small feature, threshold larger than its size
#     sic = np.zeros((10, 10))
#     sic[4, 4] = 1.0

#     lons, lats = make_lonlat_grid(10, 10)

#     edge_lons, edge_lats = extract_ice_edge_points(
#         lons, lats, sic, threshold=0.15,
#         min_feature_cells=5)

#     assert len(edge_lons) == 0
#     assert len(edge_lats) == 0


# ---------------------------------------------------------------------#
# compute_distance_to_ice_edge
# ---------------------------------------------------------------------#

def test_compute_distance_to_ice_edge_known_value():
    # one edge point at (lon=0, lat=0), one query point 1 degree
    # north of it -> distance along a meridian is exact even on
    # a spherical approximation
    edge_lons = np.array([0.0])
    edge_lats = np.array([0.0])
    pts_lons = np.array([0.0])
    pts_lats = np.array([1.0])

    dist = compute_distance_to_ice_edge(pts_lons, pts_lats,
                                        edge_lons, edge_lats)

    expected = np.deg2rad(1.0) * EARTH_RADIUS_M
    assert dist[0] == pytest.approx(expected, rel=1e-6)


def test_compute_distance_to_ice_edge_picks_nearest():
    edge_lons = np.array([0.0, 10.0])
    edge_lats = np.array([0.0, 0.0])
    pts_lons = np.array([0.5])
    pts_lats = np.array([0.0])

    dist = compute_distance_to_ice_edge(pts_lons, pts_lats,
                                        edge_lons, edge_lats)

    expected = np.deg2rad(0.5) * EARTH_RADIUS_M
    assert dist[0] == pytest.approx(expected, rel=1e-2)


def test_compute_distance_to_ice_edge_no_edge_returns_nan():
    pts_lons = np.array([0.0, 1.0])
    pts_lats = np.array([0.0, 1.0])

    dist = compute_distance_to_ice_edge(pts_lons, pts_lats,
                                        np.array([]), np.array([]))

    assert len(dist) == 2
    assert np.all(np.isnan(dist))


def test_compute_distance_to_ice_edge_zero_at_edge():
    edge_lons = np.array([5.0])
    edge_lats = np.array([5.0])
    pts_lons = np.array([5.0])
    pts_lats = np.array([5.0])

    dist = compute_distance_to_ice_edge(pts_lons, pts_lats,
                                        edge_lons, edge_lats)

    assert dist[0] == pytest.approx(0.0, abs=1.0)


# ---------------------------------------------------------------------#
# integration: fetch real ice field via model_class (ww3_4km)
# ---------------------------------------------------------------------#

# def test_get_ice_field_ww3_4km():
#     ice_lons, ice_lats, sic = get_ice_field(
#         datetime(2023, 6, 1), nID="ww3_4km", varalias="SIC")

#     assert ice_lons is not None
#     assert ice_lats is not None
#     assert sic is not None
#     assert ice_lons.shape == ice_lats.shape == sic.shape
#     assert ice_lons.ndim == 2


# def test_get_dist_to_ice_edge_ww3_4km():
#     # a couple of points in the ww3_4km domain, far enough apart
#     # that at least a finite/nan result is returned for each
#     pts_lons = np.array([5.8, 6.6])
#     pts_lats = np.array([62.3, 63.1])

#     dist = get_dist_to_ice_edge(
#         pts_lons, pts_lats, datetime(2023, 6, 1), nID="ww3_4km",
#         varalias="SIC")

#     assert len(dist) == 2
#     # either a finite, non-negative distance, or nan if there is
#     # no ice in the domain at this date/threshold
#     for d in dist:
#         assert np.isnan(d) or d >= 0.0