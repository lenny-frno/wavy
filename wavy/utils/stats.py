#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical and wave-physics utility functions.
"""

import sys
import math
import numpy as np

from wavy.logmod import get_logger

logger = get_logger(__name__)


def runmean_old(vec, win, mode=None, weights=None) -> tuple:
    """
    Computes the running mean with various configurations.

    Args:
        vec (numpy.ndarray | list): array of values to be smoothed
        win (int): window length
        mode (str): 'left', 'centered', or 'right'
        weights (numpy.ndarray | list): weights (same size as win)

    Returns:
        tuple (out, std): smoothed values and standard deviation
    """
    win = int(win)
    if mode is None:
        mode = "centered"
    out = np.zeros(len(vec)) * np.nan
    std = np.zeros(len(vec)) * np.nan
    length = len(vec) - win + 1
    if mode == "left":
        count = win - 1
        start = win - 1
        for i in range(length):
            out[count] = np.mean(vec[count - start : count + 1])
            std[count] = np.std(vec[count - start : count + 1])
            count = count + 1
    elif mode == "centered":
        start = int(math.floor(win / 2))
        for i in range(start, length):
            if win % 2 == 0:
                sys.exit("window length needs to be odd!")
            else:
                sidx = int(i - start)
                eidx = int(i + start + 1)
                if weights is not None:
                    out[i] = np.sum(vec[sidx:eidx] * weights)
                else:
                    out[i] = np.mean(vec[sidx:eidx])
                std[i] = np.std(vec[sidx:eidx])
    elif mode == "right":
        count = int(0)
        for i in range(length):
            out[count] = np.mean(vec[i : i + win])
            std[count] = np.std(vec[i : i + win])
            count = count + 1
    return out, std


def runmean(vec, win, mode=None, weights=None) -> tuple:
    """
    Computes the running mean with various configurations.

    Args:
        vec (numpy.ndarray | list): array of values to be smoothed
        win (int): window length
        mode (str): 'left', 'centered', or 'right'
        weights (numpy.ndarray | list): weights (same size as win)

    Returns:
        tuple (out, std): smoothed values and standard deviation
    """
    win = int(win)
    if mode is None:
        mode = "centered"
    out = np.zeros(len(vec)) * np.nan
    std = np.zeros(len(vec)) * np.nan
    if mode == "left":
        length = len(vec)
        start = win - 1
        for i in range(start, length):
            out[i] = np.mean(vec[i - win + 1 : i + 1])
            std[i] = np.std(vec[i - win + 1 : i + 1])
    elif mode == "centered":
        length = len(vec) - math.floor(win / 2)
        start = int(math.floor(win / 2))
        for i in range(start, length):
            if win % 2 == 0:
                sys.exit("window length needs to be odd!")
            else:
                sidx = int(i - start)
                eidx = int(i + start + 1)
                if weights is not None:
                    out[i] = np.sum(vec[sidx:eidx] * weights)
                else:
                    out[i] = np.mean(vec[sidx:eidx])
                std[i] = np.std(vec[sidx:eidx])
    elif mode == "right":
        length = len(vec)
        for i in range(length - win + 1):
            out[i] = np.mean(vec[i : i + win])
            std[i] = np.std(vec[i : i + win])
    return out, std


def runmean_conv(x: np.ndarray, win: int, mode="flat") -> np.ndarray:
    """
    Running mean using convolution.

    Args:
        x (numpy.ndarray): array of values to be smoothed
        win (int): window length
        mode (str): smoothing window type

    Notes:
        https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

    Returns:
        out (numpy.ndarray): smoothed values

    Raises:
        ValueError: for wrong dimension of x or wrong window size
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < win:
        raise ValueError("Input vector needs to be bigger than window size.")
    if win < 3:
        logger.warning("Window length too small, returning original signal")
        return x
    s = np.r_[x[win - 1 : 0 : -1], x, x[-2 : -win - 1 : -1]]
    if mode == "flat":
        w = np.ones(win, "d")
    else:
        w = eval("numpy." + mode + "(win)")  # noqa: S307
    out = np.convolve(w / w.sum(), s, mode="valid")
    return out


def bootstr(a, reps):
    """
    Conducts a simple naive bootstrap.

    Args:
        a:    time series of length n
        reps: number of repetitions

    Returns:
        tuple (b, bidx): array of draws (n x reps) and their indices
    """
    n = len(a)
    b = np.random.choice(a, (n, reps))
    bidx = np.zeros(b.shape) * np.nan
    for i in range(len(a)):
        tmp = np.where(b == a[i])
        bidx[tmp[0], tmp[1]] = i
        del tmp
    return b, bidx.astype("int")


def marginalize(a, b=None):
    """
    Remove entries in both time series that are NaN.

    Args:
        a: numpy array with np.nan for invalids
        b: optional second array
    """
    if b is None:
        return a[~np.isnan(a)]
    else:
        comb = a + b
        idx = np.array(range(len(a)))[~np.isnan(comb)]
        a1 = a[idx]
        b1 = b[idx]
        return a1, b1, idx


def compute_quantiles(ts, lq):
    """
    Compute quantiles for a given time series.

    Args:
        ts: iterable of values
        lq: iterable of quantile levels

    Returns:
        numpy array of quantiles
    """
    ts = marginalize(ts)
    return np.array([np.quantile(ts, q) for q in lq])


def dispersion_deep_water(T=None, k=None, l=None, cp=None, cg=None):
    """Compute requested variable from dispersion relation in deep water."""
    return


def dispersion_shallow_water(l=None, h=None, T=None):
    return


def dispersion_intermediate_water(l=None, h=None, T=None):
    return


def calc_deep_water_T(l=None):
    g = 9.81
    return np.sqrt(l * 2 * math.pi / g)


def calc_shallow_water_T(l=None, h=None):
    g = 9.81
    return l / np.sqrt(g * h)


def wave_length_mask_swim(ds, llim=50, ulim=2000):
    """Remove all results for wavelengths outside [llim, ulim]."""
    res = ds.where((calc_deep_water_T(ds) > llim) & (calc_deep_water_T(ds) < ulim))
    mask = ~np.isnan(res)
    return mask
