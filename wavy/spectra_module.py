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
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from sklearn.neighbors import BallTree

try:
    import wavespectra
except ImportError:
    wavespectra = None


EARTH_RADIUS_M = 6371000.0


# ---------------------------------------------------------------------#
# General helpers
# ---------------------------------------------------------------------#

def _check_wavespectra():
    """Raise a useful error if wavespectra is not installed."""
    if wavespectra is None:
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


def _normalize_spectral_dataset(ds):
    """
    Normalize environmental variable names to those expected by wavy.

    wavespectra internally expects spectral dimensions:
        freq, dir
    """
    rename = {}

    if "wnd" in ds.data_vars and "wspd" not in ds.data_vars:
        rename["wnd"] = "wspd"

    if "wnddir" in ds.data_vars and "wdir" not in ds.data_vars:
        rename["wnddir"] = "wdir"

    if rename:
        ds = ds.rename(rename)

    return ds

@lru_cache(maxsize=8)
def read_spectral_file(filename, **kwargs):
    """
    Read a native WAVEWATCH III spectral NetCDF file using wavespectra.

    The WW3 backend is required rather than the generic NetCDF reader:
    WW3 stores directional spectral density per radian and its spectral
    direction coordinate in a going-to convention. The backend converts
    density to per degree and the spectral direction coordinate to
    wavespectra's coming-from convention before the spectrum is integrated
    or passed to PTM1. Native WW3 ``wnddir`` is already a wind-from
    direction and is therefore renamed, not rotated, by the backend.

    The dataset is normalized by the backend to use:

        frequency
        direction
        site
        time
        efth
    """
    _check_wavespectra()

    logger = logging.getLogger(__name__)
    logger.debug("Reading spectral file: %s", filename)

    import xarray as xr

    unexpected = set(kwargs) - {"chunks"}
    if unexpected:
        raise TypeError(
            "read_spectral_file only accepts 'chunks' for native WW3 files; "
            f"unsupported arguments: {sorted(unexpected)}"
        )

    open_kwargs = {"engine": "ww3"}
    if kwargs.get("chunks") is not None:
        open_kwargs["chunks"] = kwargs["chunks"]

    try:
        ds = xr.open_dataset(filename, **open_kwargs)
    except Exception as exc:
        raise ValueError(
            f"Could not read '{filename}' with wavespectra's WW3 backend. "
            "This reader intentionally does not fall back to the generic "
            "NetCDF path because it would misinterpret native WW3 directional "
            "spectra."
        ) from exc

    ds = _normalize_spectral_dataset(ds)

    # ------------------------------------------------------------------
    # wavespectra should now have normalized these names.
    # ------------------------------------------------------------------
    if "efth" not in ds:
        raise KeyError(
            "wavespectra did not produce an 'efth' variable. "
            f"Available variables: {list(ds.data_vars)}"
        )

    spec = ds["efth"]

    if "freq" not in spec.dims:
        raise ValueError(
            "Spectral energy variable 'efth' has no 'freq' "
            f"dimension after wavespectra normalization. "
            f"Dimensions are: {spec.dims}"
        )

    if "dir" not in spec.dims:
        raise ValueError(
            "Spectral energy variable 'efth' has no 'dir' "
            f"dimension after wavespectra normalization. "
            f"Dimensions are: {spec.dims}"
        )

    if "freq" not in spec.coords:
        raise ValueError(
            "The 'efth' DataArray has a freq dimension but no "
            "freq coordinate."
        )

    if "dir" not in spec.coords:
        raise ValueError(
            "The 'efth' DataArray has a dir dimension but no "
            "dir coordinate."
        )

    logger.debug(
        "Read spectral dataset: dimensions=%s, variables=%s",
        ds.dims,
        list(ds.data_vars),
    )

    return ds


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

    def query_many(self, lons, lats):
        """
        Find nearest spectral points for many locations at once.

        Parameters
        ----------
        lons, lats : array-like

        Returns
        -------
        indices : numpy.ndarray
        distances_m : numpy.ndarray
        """
        query = np.deg2rad(
            np.column_stack((np.asarray(lats, dtype=float), np.asarray(lons, dtype=float)))
        )

        distance, index = self.tree.query(query, k=1)

        return index[:, 0].astype(int), distance[:, 0] * EARTH_RADIUS_M


def _resolve_point_dim(ds, point_dim=None):
    """Resolve spectral point dimension name once."""
    if point_dim is not None:
        return point_dim

    for candidate in [
        "site",
        "station",
        "point",
        "node",
        "location",
    ]:
        if candidate in ds.dims:
            return candidate

    raise KeyError(
        "Could not identify spectral point dimension. "
        "Specify point_dim explicitly."
    )


def _resolve_time_name(ds, time_name=None):
    """Resolve spectral time coordinate/dimension name once."""
    if time_name is not None:
        return time_name

    for candidate in ["time", "datetime", "valid_time"]:
        if candidate in ds.coords:
            return candidate

    for candidate in ["time", "datetime", "valid_time"]:
        if candidate in ds.dims:
            return candidate

    raise KeyError(
        "Could not identify spectral time coordinate. "
        "Specify time_name explicitly."
    )


def _nearest_time_lookup(spectral_times, target_times):
    """
    Vectorized nearest-neighbour lookup for datetime64[ns] arrays.

    Parameters
    ----------
    spectral_times : numpy.ndarray
        Available spectral times, datetime64[ns].
    target_times : numpy.ndarray
        Target times, datetime64[ns].

    Returns
    -------
    indices : numpy.ndarray
    matched_times : numpy.ndarray
    differences_seconds : numpy.ndarray
    """
    spectral_times = np.asarray(spectral_times).astype("datetime64[ns]")
    target_times = np.asarray(target_times).astype("datetime64[ns]")

    if spectral_times.size == 0:
        raise ValueError("Spectral Dataset contains no time values.")

    order = np.argsort(spectral_times)
    sorted_times = spectral_times[order]

    right = np.searchsorted(sorted_times, target_times, side="left")
    right = np.clip(right, 0, len(sorted_times) - 1)
    left = np.clip(right - 1, 0, len(sorted_times) - 1)

    right_diff = np.abs(sorted_times[right] - target_times)
    left_diff = np.abs(sorted_times[left] - target_times)

    choose_left = left_diff <= right_diff
    sorted_indices = np.where(choose_left, left, right)

    indices = order[sorted_indices].astype(int)
    matched_times = spectral_times[indices]
    differences = np.abs(matched_times - target_times)
    differences_seconds = differences.astype("timedelta64[s]").astype(float)

    return indices, matched_times, differences_seconds


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
    indices, matched_times, differences = _nearest_time_lookup(
        times,
        np.asarray([target_time]),
    )

    return int(indices[0]), matched_times[0], float(differences[0])


# ---------------------------------------------------------------------#
# Spectral extraction
# ---------------------------------------------------------------------#
def compute_spectrum_peak_frequency(spectrum):
    """
    Compute peak frequency (Hz) of the whole 2D spectrum.

    The whole spectrum is reduced to 1D by summing efth over direction,
    then selecting the frequency with maximum energy.
    """
    if "efth" not in spectrum:
        raise KeyError("Spectrum Dataset must contain 'efth'.")

    efth = spectrum["efth"]

    if "freq" not in efth.dims or "dir" not in efth.dims:
        raise ValueError(
            "Spectrum 'efth' must have (freq, dir) dimensions to compute "
            f"peak frequency. Got {efth.dims}"
        )

    if "freq" not in efth.coords:
        raise ValueError("Spectrum 'efth' has no 'freq' coordinate.")

    energy_1d = efth.sum(dim="dir", skipna=True)
    values = np.asarray(energy_1d.values).squeeze()
    freqs = np.asarray(energy_1d["freq"].values).squeeze()

    if values.ndim != 1 or freqs.ndim != 1:
        raise ValueError(
            "Peak-frequency computation expects 1D arrays after reducing direction."
        )

    mask = np.isfinite(values) & np.isfinite(freqs)
    if not np.any(mask):
        return np.nan

    values_valid = values[mask]
    freqs_valid = freqs[mask]
    idx = int(np.nanargmax(values_valid))
    return float(freqs_valid[idx])

def compute_spectrum_direction_diagnostics(spectrum, partitioned=None):
    """
    Compute wind direction, whole-spectrum mean direction, and
    per-partition mean direction.

    Returns
    -------
    dict
        spectral_wind_dir_deg : float
        spectral_mean_direction_deg : float
        partition_mean_direction_deg : numpy.ndarray
    """
    if "efth" not in spectrum:
        raise KeyError("Spectrum Dataset must contain 'efth'.")

    spectral_wind_dir_deg = np.nan
    if "wdir" in spectrum:
        wdir_values = np.asarray(spectrum["wdir"].values).squeeze()
        if np.size(wdir_values) > 0 and np.any(np.isfinite(wdir_values)):
            spectral_wind_dir_deg = float(
                np.ravel(wdir_values)[np.flatnonzero(np.isfinite(wdir_values))[0]]
            )

    stats = spectrum.spec.stats(["dm"])
    dm_values = np.asarray(stats["dm"].values).squeeze()
    if np.size(dm_values) == 0 or not np.any(np.isfinite(dm_values)):
        spectral_mean_direction_deg = np.nan
    else:
        spectral_mean_direction_deg = float(
            np.ravel(dm_values)[np.flatnonzero(np.isfinite(dm_values))[0]]
        )

    partition_mean_direction_deg = np.array([], dtype=float)
    if partitioned is not None:
        part_stats = partitioned.spec.stats(["dm"])
        part_dm = np.asarray(part_stats["dm"].values).squeeze()
        part_dm = np.atleast_1d(part_dm).astype(float)
        partition_mean_direction_deg = part_dm

    return {
        "spectral_wind_dir_deg": spectral_wind_dir_deg,
        "spectral_mean_direction_deg": spectral_mean_direction_deg,
        "partition_mean_direction_deg": partition_mean_direction_deg,
    }

def extract_point_spectrum(
    ds,
    point_index,
    time_index,
    point_dim=None,
    time_name=None,
):
    """
    Extract one spectrum from an unstructured spectral Dataset.

    Returns a Dataset containing one spectrum with dimensions:
        freq, dir
    """

    point_dim = _resolve_point_dim(ds, point_dim=point_dim)
    time_name = _resolve_time_name(ds, time_name=time_name)

    # Select the spectrum.
    spectrum = ds.isel(
        {
            point_dim: point_index,
            time_name: time_index,
        }
    )

    # Remove singleton dimensions created by the selection.
    spectrum = spectrum.squeeze(drop=True)

    # ------------------------------------------------------------------
    # Keep only what is needed by wavespectra.
    # ------------------------------------------------------------------
    if "efth" not in spectrum:
        raise KeyError(
            "Spectral Dataset does not contain 'efth'."
        )

    efth = spectrum["efth"]

    if "freq" not in efth.dims or "dir" not in efth.dims:
        raise ValueError(
            "Extracted spectrum does not have the expected "
            f"(freq, dir) dimensions. Got {efth.dims}"
        )

    if "freq" not in efth.coords:
        raise ValueError(
            "Extracted spectrum has no 'freq' coordinate."
        )

    if "dir" not in efth.coords:
        raise ValueError(
            "Extracted spectrum has no 'dir' coordinate."
        )

    return spectrum


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
        Single-point, single-time spectral Dataset.
    method : str
        wavespectra partitioning method, e.g. 'ptm1', 'ptm2'.

    Returns
    -------
    xarray.Dataset
        Partitioned spectrum.
    """

    if "efth" not in spectrum:
        raise KeyError(
            "Spectrum Dataset must contain 'efth'."
        )

    efth = spectrum["efth"]

    if "freq" not in efth.dims:
        raise ValueError(
            f"'efth' must have a freq dimension, got {efth.dims}"
        )

    if "dir" not in efth.dims:
        raise ValueError(
            f"'efth' must have a dir dimension, got {efth.dims}"
        )

    # Access the wavespectra partition namespace.
    partition = efth.spec.partition

    method = method.lower()

    if not hasattr(partition, method):
        raise ValueError(
            f"Unknown wavespectra partition method '{method}'. "
            f"Available methods include: "
            f"ptm1, ptm2, ptm3, ptm4, ptm5, hp01, bbox."
        )

    partition_method = getattr(partition, method)

    # ---------------------------------------------------------------
    # PTM methods require environmental fields.
    # ---------------------------------------------------------------
    if method in ("ptm1", "ptm2", "ptm1_track"):
        required = ["wspd", "wdir", "dpt"]

        missing = [
            name for name in required
            if name not in spectrum
        ]

        if missing:
            raise KeyError(
                f"Partition method '{method}' requires the following "
                f"variables in the spectral Dataset: {missing}. "
                f"Available variables: {list(spectrum.data_vars)}"
            )

        return partition_method(
            wspd=spectrum["wspd"],
            wdir=spectrum["wdir"],
            dpt=spectrum["dpt"],
            **kwargs,
        )

    return partition_method(**kwargs)

# ---------------------------------------------------------------------#
# Wave-regime classification and diagnostics
# ---------------------------------------------------------------------#

def compute_wave_regime_diagnostics(
    partitioned,
    wind_sea_threshold=0.75,
    swell_dominated_threshold=0.25,
    min_partition_contribution=0.05,
    hs_calm=None,
):
    """
    Compute wave-regime diagnostics from a partitioned spectrum.

    Assumes PTM1 partition ordering: part=0 is wind sea, the
    remaining partitions are swells sorted by descending Hs.

    Parameters
    ----------
    partitioned : xarray.Dataset
        Output of partition_spectrum.
    wind_sea_threshold : float
        Wind-sea energy fraction at/above which the regime is
        wind-sea dominated.
    swell_dominated_threshold : float
        Wind-sea energy fraction at/below which the regime is
        swell dominated.
    min_partition_contribution : float
        Minimum fraction of total energy a partition must carry to
        be counted as a distinct wave system (filters watershed
        noise partitions).
    hs_calm : float, optional
        Total Hs below which conditions are considered calm and
        n_wave_systems is forced to zero. If None, no calm check
        is applied.

    Returns
    -------
    dict
        wave_regime : int or nan
            0 = wind-sea dominated, 1 = mixed, 2 = swell dominated.
        wind_sea_fraction : float
            Energy fraction attributed to the wind-sea partition.
        n_wave_systems : int or nan
            Number of partitions carrying at least
            min_partition_contribution of total energy (0 if calm).
        hs_total : float
            Combined significant wave height across all partitions.
    """
    stats = partitioned.spec.stats(["hs"])
    hs = np.asarray(stats["hs"].values).squeeze()
    hs = hs[np.isfinite(hs)]

    nan_result = {
        "wave_regime": np.nan,
        "wind_sea_fraction": np.nan,
        "n_wave_systems": np.nan,
        "hs_total": np.nan,
    }

    if hs.size == 0:
        return nan_result

    energy = hs ** 2
    total_energy = np.sum(energy)

    if total_energy <= 0:
        return nan_result

    hs_total = float(np.sqrt(total_energy))
    wind_sea_fraction = float(energy[0] / total_energy)

    if wind_sea_fraction >= wind_sea_threshold:
        wave_regime = 0
    elif wind_sea_fraction <= swell_dominated_threshold:
        wave_regime = 2
    else:
        wave_regime = 1

    energy_fraction = energy / total_energy
    n_wave_systems = int(np.sum(energy_fraction >= min_partition_contribution))

    if hs_calm is not None and hs_total < hs_calm:
        n_wave_systems = 0

    return {
        "wave_regime": wave_regime,
        "wind_sea_fraction": wind_sea_fraction,
        "n_wave_systems": n_wave_systems,
        "hs_total": hs_total,
    }


def classify_wave_regime(
    partitioned,
    wind_sea_threshold=0.75,
    swell_dominated_threshold=0.25,
):
    """Kept for backwards compatibility; prefer
    compute_wave_regime_diagnostics for new code."""
    return compute_wave_regime_diagnostics(
        partitioned,
        wind_sea_threshold=wind_sea_threshold,
        swell_dominated_threshold=swell_dominated_threshold,
    )["wave_regime"]


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
            "wind_sea_fraction": np.nan,
            "n_wave_systems": np.nan,
            "hs_total": np.nan,
            "peak_frequency_hz": np.nan,
            "spectral_wind_dir_deg": np.nan,
            "spectral_mean_direction_deg": np.nan,
            "partition_mean_direction_deg": np.array([], dtype=float),
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

    peak_frequency_hz = compute_spectrum_peak_frequency(spectrum)

    partitioned = partition_spectrum(
        spectrum,
        method=partition_method,
        **(partition_kwargs or {}),
    )

    direction_diagnostics = compute_spectrum_direction_diagnostics(
        spectrum,
        partitioned=partitioned,
    )

    diagnostics = compute_wave_regime_diagnostics(
        partitioned,
        **(regime_kwargs or {}),
    )

    return {
        **diagnostics,
        "peak_frequency_hz": peak_frequency_hz,
        **direction_diagnostics,
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

    Notes
    -----
    Optional keyword:
        return_profiling : bool
            If True, include a profiling dictionary under key
            "profiling" in the returned result.
    """
    logger = logging.getLogger(__name__)

    kwargs = dict(kwargs)
    allowed_kwargs = {
        "lon_name",
        "lat_name",
        "time_name",
        "point_dim",
        "max_time_difference",
        "partition_method",
        "partition_kwargs",
        "regime_kwargs",
        "return_profiling",
    }
    unexpected = set(kwargs) - allowed_kwargs
    if unexpected:
        raise TypeError(
            "collocate_spectra got unsupported keyword arguments: "
            f"{sorted(unexpected)}"
        )

    return_profiling = bool(kwargs.pop("return_profiling", False))

    lons = np.asarray(lons)
    lats = np.asarray(lats)
    times = np.asarray(times).astype("datetime64[ns]")

    if not (
        len(lons) == len(lats) == len(times)
    ):
        raise ValueError(
            "lons, lats and times must have identical lengths."
        )

    t0 = perf_counter()

    lon_name = kwargs.get("lon_name")
    lat_name = kwargs.get("lat_name")
    point_dim = _resolve_point_dim(ds, point_dim=kwargs.get("point_dim"))
    time_name = _resolve_time_name(ds, time_name=kwargs.get("time_name"))

    max_time_difference = kwargs.get("max_time_difference")
    partition_method = kwargs.get("partition_method", "ptm1")
    partition_kwargs = kwargs.get("partition_kwargs") or {}
    regime_kwargs = kwargs.get("regime_kwargs") or {}

    spectral_lons, spectral_lats = get_spectral_coordinates(
        ds,
        lon_name=lon_name,
        lat_name=lat_name,
    )

    point_indexer = SpectralPointIndex(
        spectral_lons,
        spectral_lats,
    )

    npoints = len(lons)

    wave_regime = np.full(npoints, np.nan)
    wind_sea_fraction = np.full(npoints, np.nan)
    n_wave_systems = np.full(npoints, np.nan)
    hs_total = np.full(npoints, np.nan)
    point_index = np.full(npoints, -1, dtype=int)
    distance_m = np.full(npoints, np.nan)
    time_difference_s = np.full(npoints, np.nan)
    peak_frequency_hz = np.full(npoints, np.nan)
    spectral_wind_dir_deg = np.full(npoints, np.nan)
    spectral_mean_direction_deg = np.full(npoints, np.nan)
    partition_mean_direction_deg = np.empty(npoints, dtype=object)
    partition_mean_direction_deg[:] = [
        np.array([], dtype=float) for _ in range(npoints)
    ]

    spectral_times = np.empty(npoints, dtype="datetime64[ns]")

    if npoints == 0:
        result: dict[str, Any] = {
            "wave_regime": wave_regime,
            "wind_sea_fraction": wind_sea_fraction,
            "n_wave_systems": n_wave_systems,
            "hs_total": hs_total,
            "spectral_point_index": point_index,
            "spectral_distance_m": distance_m,
            "spectral_time": spectral_times,
            "spectral_time_difference_s": time_difference_s,
            "peak_frequency_hz": peak_frequency_hz,
            "spectral_wind_dir_deg": spectral_wind_dir_deg,
            "spectral_mean_direction_deg": spectral_mean_direction_deg,
            "partition_mean_direction_deg": partition_mean_direction_deg,
        }
        if return_profiling:
            result["profiling"] = {
                "n_observations": 0,
                "n_unique_target_times": 0,
                "n_unique_point_time_pairs": 0,
                "t_total_s": 0.0,
                "t_spatial_lookup_s": 0.0,
                "t_time_lookup_s": 0.0,
                "t_partition_and_regime_s": 0.0,
            }
        return result

    t_spatial_start = perf_counter()

    finite_coords = np.isfinite(lons) & np.isfinite(lats)
    if np.any(finite_coords):
        indices, distances = point_indexer.query_many(
            lons[finite_coords],
            lats[finite_coords],
        )
        point_index[finite_coords] = indices
        distance_m[finite_coords] = distances

    t_spatial_end = perf_counter()

    t_time_start = perf_counter()

    available_times = get_spectral_times(ds, time_name=time_name)
    time_index, matched_times, time_differences = _nearest_time_lookup(
        available_times,
        times,
    )

    spectral_times[:] = matched_times
    time_difference_s[:] = time_differences

    t_time_end = perf_counter()

    valid = finite_coords & (point_index >= 0)
    if max_time_difference is not None:
        valid = valid & (time_difference_s <= max_time_difference)

    t_partition_start = perf_counter()

    if np.any(valid):
        valid_indices = np.where(valid)[0]

        pair_matrix = np.column_stack(
            (time_index[valid_indices], point_index[valid_indices])
        )
        unique_pairs, inverse = np.unique(pair_matrix, axis=0, return_inverse=True)

        regime_by_pair = np.full(len(unique_pairs), np.nan)
        wind_fraction_by_pair = np.full(len(unique_pairs), np.nan)
        n_systems_by_pair = np.full(len(unique_pairs), np.nan)
        hs_total_by_pair = np.full(len(unique_pairs), np.nan)
        peak_frequency_by_pair = np.full(len(unique_pairs), np.nan)
        spectral_wind_dir_by_pair = np.full(len(unique_pairs), np.nan)
        spectral_mean_direction_by_pair = np.full(len(unique_pairs), np.nan)
        partition_mean_direction_by_pair = np.empty(len(unique_pairs), dtype=object)
        partition_mean_direction_by_pair[:] = [
            np.array([], dtype=float) for _ in range(len(unique_pairs))
        ]
        for pair_i, (ti, pi) in enumerate(unique_pairs):
            logger.debug(
                "Spectral partitioning for unique pair %d/%d (time=%d, point=%d)",
                pair_i + 1,
                len(unique_pairs),
                ti,
                pi,
            )

            spectrum = extract_point_spectrum(
                ds,
                point_index=int(pi),
                time_index=int(ti),
                point_dim=point_dim,
                time_name=time_name,
            )

            peak_frequency_by_pair[pair_i] = compute_spectrum_peak_frequency(spectrum)

            partitioned = partition_spectrum(
                spectrum,
                method=partition_method,
                **partition_kwargs,
            )
            direction_diagnostics = compute_spectrum_direction_diagnostics(
                spectrum,
                partitioned=partitioned,
            )

            diagnostics = compute_wave_regime_diagnostics(
                partitioned,
                **regime_kwargs,
            )

            spectral_wind_dir_by_pair[pair_i] = (
                direction_diagnostics["spectral_wind_dir_deg"]
            )
            spectral_mean_direction_by_pair[pair_i] = (
                direction_diagnostics["spectral_mean_direction_deg"]
            )
            partition_mean_direction_by_pair[pair_i] = (
                direction_diagnostics["partition_mean_direction_deg"]
            )

            regime_by_pair[pair_i] = diagnostics["wave_regime"]
            wind_fraction_by_pair[pair_i] = diagnostics["wind_sea_fraction"]
            n_systems_by_pair[pair_i] = diagnostics["n_wave_systems"]
            hs_total_by_pair[pair_i] = diagnostics["hs_total"]

        mapped_pair_idx = inverse
        wave_regime[valid_indices] = regime_by_pair[mapped_pair_idx]
        wind_sea_fraction[valid_indices] = wind_fraction_by_pair[mapped_pair_idx]
        n_wave_systems[valid_indices] = n_systems_by_pair[mapped_pair_idx]
        hs_total[valid_indices] = hs_total_by_pair[mapped_pair_idx]
        peak_frequency_hz[valid_indices] = peak_frequency_by_pair[mapped_pair_idx]
        spectral_wind_dir_deg[valid_indices] = (
            spectral_wind_dir_by_pair[mapped_pair_idx]
        )
        spectral_mean_direction_deg[valid_indices] = (
            spectral_mean_direction_by_pair[mapped_pair_idx]
        )
        partition_mean_direction_deg[valid_indices] = (
            partition_mean_direction_by_pair[mapped_pair_idx]
        )
    else:
        unique_pairs = np.empty((0, 2), dtype=int)

    t_partition_end = perf_counter()

    result: dict[str, Any] = {
        "wave_regime": wave_regime,
        "wind_sea_fraction": wind_sea_fraction,
        "n_wave_systems": n_wave_systems,
        "hs_total": hs_total,
        "spectral_point_index": point_index,
        "spectral_distance_m": distance_m,
        "spectral_time": spectral_times,
        "spectral_time_difference_s": time_difference_s,
        "peak_frequency_hz": peak_frequency_hz,
        "spectral_wind_dir_deg": spectral_wind_dir_deg,
        "spectral_mean_direction_deg": spectral_mean_direction_deg,
        "partition_mean_direction_deg": partition_mean_direction_deg,
    }

    if return_profiling:
        result["profiling"] = {
            "n_observations": int(npoints),
            "n_unique_target_times": int(len(np.unique(times))),
            "n_unique_point_time_pairs": int(len(unique_pairs)),
            "t_total_s": float(perf_counter() - t0),
            "t_spatial_lookup_s": float(t_spatial_end - t_spatial_start),
            "t_time_lookup_s": float(t_time_end - t_time_start),
            "t_partition_and_regime_s": float(t_partition_end - t_partition_start),
        }

    return result


# ---------------------------------------------------------------------#