#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------#
"""
Utilities for reading and collocating unstructured wave spectral output.

The module provides:
    - reading of spectral files through wavespectra
    - nearest spectral-point lookup
    - nearest spectral-time lookup
    - spectral partitioning
    - wave-regime classification

The module is deliberately independent of collocation_class.  The
collocation layer is responsible for passing lon/lat/time and assigning
the returned values to its xarray Dataset.
"""
# ---------------------------------------------------------------------#

import logging
from copy import deepcopy
from functools import lru_cache

import numpy as np
import pandas as pd

from sklearn.neighbors import BallTree

try:
    from wavespectra import read_netcdf
except ImportError:
    read_netcdf = None


EARTH_RADIUS_M = 6371000.0


# ---------------------------------------------------------------------#
# General helpers
# ---------------------------------------------------------------------#

def _check_wavespectra():
    """Raise a useful error if wavespectra is not installed."""
    if read_netcdf is None:
        raise ImportError(
            "wavespectra is required for spectral processing. "
            "Install it with `pip install wavespectra`."
        )


def _to_datetime64_ns(value):
    """
    Convert datetime-like input to numpy datetime64[ns].

    Parameters
    ----------
    value : datetime-like

    Returns
    -------
    numpy.datetime64
    """
    return np.datetime64(pd.Timestamp(value).to_datetime64(), "ns")


def _datetime_difference_seconds(times, target):
    """
    Return absolute time difference in seconds.

    Handles numpy datetime64 arrays.
    """
    times = np.asarray(times).astype("datetime64[ns]")
    target = _to_datetime64_ns(target)

    return np.abs(times - target).astype("timedelta64[s]").astype(float)


# ---------------------------------------------------------------------#
# Spectral file reader
# ---------------------------------------------------------------------#

@lru_cache(maxsize=8)
def read_spectral_file(filename, **kwargs):
    """
    Read a spectral file using wavespectra.

    The result is cached because the same spectral file will normally
    be queried many times during one collocation operation.

    Parameters
    ----------
    filename : str
        Path to spectral NetCDF file.

    Returns
    -------
    xarray.Dataset
        Spectral dataset returned by wavespectra.
    """
    _check_wavespectra()

    logger = logging.getLogger(__name__)
    logger.debug("Reading spectral file: %s", filename)

    return read_netcdf(filename, **kwargs)


# ---------------------------------------------------------------------#
# Coordinate discovery
# ---------------------------------------------------------------------#

def get_spectral_coordinates(ds, lon_name=None, lat_name=None):
    """
    Return longitude and latitude arrays from a spectral Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
    lon_name, lat_name : str, optional
        Explicit coordinate names.

    Returns
    -------
    lons, lats : numpy.ndarray
    """
    if lon_name is None:
        for candidate in ["lon", "longitude", "lons"]:
            if candidate in ds:
                lon_name = candidate
                break

    if lat_name is None:
        for candidate in ["lat", "latitude", "lats"]:
            if candidate in ds:
                lat_name = candidate
                break

    if lon_name is None or lat_name is None:
        raise KeyError(
            "Could not identify spectral longitude/latitude variables. "
            "Specify lon_name and lat_name explicitly."
        )

    # Spectral stations are stationary. If coordinates have a time
    # dimension, use the first time step and remove that dimension.
    lon = ds[lon_name]
    lat = ds[lat_name]

    if "time" in lon.dims:
        lon = lon.isel(time=0, drop=True)

    if "time" in lat.dims:
        lat = lat.isel(time=0, drop=True)

    lons = np.asarray(lon.values).squeeze()
    lats = np.asarray(lat.values).squeeze()

    if lons.ndim != 1 or lats.ndim != 1:
        raise ValueError(
            "Spectral point coordinates must be one-dimensional after "
            "removing the time dimension."
        )

    if len(lons) != len(lats):
        raise ValueError(
            "Spectral longitude and latitude arrays have different lengths."
        )

    return lons, lats


def get_spectral_times(ds, time_name=None):
    """
    Return spectral time coordinate.

    Parameters
    ----------
    ds : xarray.Dataset
    time_name : str, optional

    Returns
    -------
    numpy.ndarray
        Datetime64 array where possible.
    """
    if time_name is None:
        for candidate in ["time", "datetime", "valid_time"]:
            if candidate in ds.coords:
                time_name = candidate
                break

    if time_name is None:
        raise KeyError(
            "Could not identify spectral time coordinate. "
            "Specify time_name explicitly."
        )

    values = ds[time_name].values

    try:
        return np.asarray(values).astype("datetime64[ns]")
    except (TypeError, ValueError):
        return np.asarray(
            [np.datetime64(pd.Timestamp(v)) for v in values]
        )


# ---------------------------------------------------------------------#
# Nearest-neighbour index
# ---------------------------------------------------------------------#

class SpectralPointIndex:
    """
    Cached nearest-neighbour index for unstructured spectral points.

    The BallTree uses radians and the haversine metric, giving a true
    great-circle distance rather than treating lon/lat as Cartesian
    coordinates.
    """

    def __init__(self, lons, lats):
        self.lons = np.asarray(lons)
        self.lats = np.asarray(lats)

        coordinates = np.deg2rad(
            np.column_stack((self.lats, self.lons))
        )

        self.tree = BallTree(
            coordinates,
            metric="haversine"
        )

    def query(self, lon, lat):
        """
        Find nearest spectral point.

        Returns
        -------
        index : int
        distance_m : float
        """
        query = np.deg2rad(
            np.array([[float(lat), float(lon)]])
        )

        distance, index = self.tree.query(query, k=1)

        return int(index[0, 0]), float(distance[0, 0] * EARTH_RADIUS_M)


# ---------------------------------------------------------------------#
# Spectral time selection
# ---------------------------------------------------------------------#

def find_nearest_spectral_time(ds, target_time, time_name=None):
    """
    Find nearest available spectral time.

    Returns
    -------
    index : int
    time : numpy.datetime64
    difference_seconds : float
    """
    times = get_spectral_times(ds, time_name=time_name)

    differences = _datetime_difference_seconds(times, target_time)

    index = int(np.argmin(differences))

    return (
        index,
        times[index],
        float(differences[index])
    )


# ---------------------------------------------------------------------#
# Spectral extraction
# ---------------------------------------------------------------------#

def extract_point_spectrum(
    ds,
    point_index,
    time_index,
    point_dim=None,
    time_name=None,
):
    """
    Extract one spectrum from an unstructured spectral Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
    point_index : int
        Index of nearest spectral point.
    time_index : int
        Index of nearest spectral time.
    point_dim : str, optional
        Dimension containing spectral points.
    time_name : str, optional

    Returns
    -------
    xarray.Dataset
        One-point, one-time spectral Dataset.
    """
    if point_dim is None:
        candidates = [
            "site",
            "station",
            "point",
            "node",
            "location",
        ]

        for candidate in candidates:
            if candidate in ds.dims:
                point_dim = candidate
                break

    if point_dim is None:
        raise KeyError(
            "Could not identify spectral point dimension. "
            "Specify point_dim explicitly."
        )

    if time_name is None:
        for candidate in ["time", "datetime", "valid_time"]:
            if candidate in ds.dims:
                time_name = candidate
                break

    if time_name is None:
        raise KeyError(
            "Could not identify spectral time dimension."
        )

    return ds.isel(
        {
            point_dim: point_index,
            time_name: time_index,
        }
    )


# ---------------------------------------------------------------------#
# Partitioning
# ---------------------------------------------------------------------#

def partition_spectrum(
    spectrum,
    method="ptm1",
    **kwargs,
):
    """
    Partition a spectrum using wavespectra.

    Parameters
    ----------
    spectrum : xarray.Dataset
        Single-point spectrum.
    method : str
        wavespectra partitioning method.

    Returns
    -------
    xarray.Dataset
        Partitioned spectrum.
    """
    # wavespectra exposes partitioning through the SpecArray accessor.
    #
    # Keep this wrapper deliberately small so the exact partitioning
    # algorithm can be changed through configuration without affecting
    # the collocation code.

    if not hasattr(spectrum, "spec"):
        raise AttributeError(
            "The supplied Dataset does not expose the wavespectra "
            "`spec` accessor."
        )

    spec = spectrum.spec

    if not hasattr(spec, "partition"):
        raise AttributeError(
            "Installed wavespectra version does not expose "
            "`spec.partition`."
        )

    return spec.partition(
        method=method,
        **kwargs
    )


# ---------------------------------------------------------------------#
# Wave-regime classification
# ---------------------------------------------------------------------#

def classify_wave_regime(
    partitioned,
    wind_sea_threshold=0.5,
    swell_threshold=0.5,
    mixed_threshold=0.2,
):
    """
    Classify a partitioned spectrum into a simple wave regime.

    IMPORTANT:
        The exact regime definition is project-specific.  This function
        therefore provides a transparent default which should be replaced
        or configured if a scientific definition already exists.

    Returns
    -------
    int
        Regime code.

    Notes
    -----
    Suggested codes:

        0 = wind-sea dominated
        1 = mixed
        2 = swell dominated
    """
    # This function intentionally does not assume a particular wavespectra
    # partition Dataset layout.  The classification should be implemented
    # against the variables produced by the selected partitioning method.

    raise NotImplementedError(
        "Implement the project-specific wave-regime classification "
        "against the output of the selected wavespectra partitioning "
        "method."
    )


# ---------------------------------------------------------------------#
# Single collocation
# ---------------------------------------------------------------------#

def collocate_spectrum(
    ds,
    lon,
    lat,
    time,
    point_indexer=None,
    lon_name=None,
    lat_name=None,
    time_name=None,
    point_dim=None,
    max_time_difference=None,
    partition_method="ptm1",
    partition_kwargs=None,
    regime_kwargs=None,
):
    """
    Collocate one arbitrary lon/lat/time position to spectral output.

    Parameters
    ----------
    ds : xarray.Dataset
        Spectral Dataset.
    lon, lat : float
        Target position.
    time : datetime-like
        Target time.
    point_indexer : SpectralPointIndex, optional
        Reusable nearest-neighbour index.
    max_time_difference : float, optional
        Maximum allowed time difference in seconds.
    partition_method : str
        wavespectra partitioning method.
    partition_kwargs : dict, optional
        Arguments passed to partition_spectrum.
    regime_kwargs : dict, optional
        Arguments passed to classify_wave_regime.

    Returns
    -------
    dict
        Collocation information and wave regime.
    """
    logger = logging.getLogger(__name__)

    if point_indexer is None:
        lons, lats = get_spectral_coordinates(
            ds,
            lon_name=lon_name,
            lat_name=lat_name,
        )

        point_indexer = SpectralPointIndex(lons, lats)

    point_index, distance_m = point_indexer.query(lon, lat)

    time_index, matched_time, time_difference = \
        find_nearest_spectral_time(
            ds,
            time,
            time_name=time_name,
        )

    if (
        max_time_difference is not None
        and time_difference > max_time_difference
    ):
        logger.warning(
            "Nearest spectral time is %.1f seconds away from target "
            "time; maximum allowed difference is %.1f seconds.",
            time_difference,
            max_time_difference,
        )

        return {
            "wave_regime": np.nan,
            "spectral_point_index": point_index,
            "spectral_distance_m": distance_m,
            "spectral_time": matched_time,
            "spectral_time_difference_s": time_difference,
        }

    spectrum = extract_point_spectrum(
        ds,
        point_index=point_index,
        time_index=time_index,
        point_dim=point_dim,
        time_name=time_name,
    )

    partitioned = partition_spectrum(
        spectrum,
        method=partition_method,
        **(partition_kwargs or {}),
    )

    wave_regime = classify_wave_regime(
        partitioned,
        **(regime_kwargs or {}),
    )

    return {
        "wave_regime": wave_regime,
        "spectral_point_index": point_index,
        "spectral_distance_m": distance_m,
        "spectral_time": matched_time,
        "spectral_time_difference_s": time_difference,
    }


# ---------------------------------------------------------------------#
# Track collocation
# ---------------------------------------------------------------------#

def collocate_spectra(
    ds,
    lons,
    lats,
    times,
    **kwargs,
):
    """
    Collocate a complete track against an unstructured spectral Dataset.

    The BallTree is constructed exactly once and reused for all points.

    Parameters
    ----------
    ds : xarray.Dataset
    lons, lats : array-like
        Track positions.
    times : array-like
        Track times.

    Returns
    -------
    dict of numpy.ndarray
        One-dimensional collocation results.
    """
    logger = logging.getLogger(__name__)

    lons = np.asarray(lons)
    lats = np.asarray(lats)
    times = np.asarray(times)# Observation times used for spectral collocation
    obs_time = pd.to_datetime(new.vars['obs_time'].values)
    unique_times = pd.unique(new.vars['model_time'].values)

    if not (
        len(lons) == len(lats) == len(times)
    ):
        raise ValueError(
            "lons, lats and times must have identical lengths."
        )

    spectral_lons, spectral_lats = get_spectral_coordinates(
        ds,
        lon_name=kwargs.get("lon_name"),
        lat_name=kwargs.get("lat_name"),
    )

    point_indexer = SpectralPointIndex(
        spectral_lons,
        spectral_lats,
    )

    npoints = len(lons)

    wave_regime = np.full(npoints, np.nan)
    point_index = np.full(npoints, -1, dtype=int)
    distance_m = np.full(npoints, np.nan)
    time_difference_s = np.full(npoints, np.nan)

    spectral_times = np.empty(
        npoints,
        dtype="datetime64[ns]",
    )

    for i in range(npoints):
        logger.debug(
            "Spectral collocation %d/%d",
            i + 1,
            npoints,
        )

        result = collocate_spectrum(
            ds,
            lon=lons[i],
            lat=lats[i],
            time=times[i],
            point_indexer=point_indexer,
            **kwargs,
        )

        wave_regime[i] = result["wave_regime"]
        point_index[i] = result["spectral_point_index"]
        distance_m[i] = result["spectral_distance_m"]
        time_difference_s[i] = result[
            "spectral_time_difference_s"
        ]
        spectral_times[i] = result["spectral_time"]

    return {
        "wave_regime": wave_regime,
        "spectral_point_index": point_index,
        "spectral_distance_m": distance_m,
        "spectral_time": spectral_times,
        "spectral_time_difference_s": time_difference_s,
    }


# ---------------------------------------------------------------------#