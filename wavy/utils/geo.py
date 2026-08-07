#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geospatial and distance utility functions.
"""

import numpy as np
from math import radians, cos, sin, asin, sqrt

from wavy.utils.misc import get_item_child


def haversineP(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    km = 6367 * c
    return km


def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points on the earth.
    All args must be of equal length.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def haversineA(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points on the earth.
    lon1, lat1, lon2, lat2 can be scalars or lists.
    """
    rads = np.deg2rad(np.array([lon1, lat1, lon2, lat2]))
    if isinstance(lon1, list):
        dlon = rads[2, :] - rads[0, :]
        dlat = rads[3, :] - rads[1, :]
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(rads[1, :]) * np.cos(rads[3, :]) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6367 * c
        return list(km)
    else:
        dlon = rads[2] - rads[0]
        dlat = rads[3] - rads[1]
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(rads[1]) * np.cos(rads[3]) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6367 * c
        return [km]


def footprint_pulse_limited_radius(Hs: float, h: float, tau: float) -> float:
    """
    Pulse limited footprint radius according to Chelton et al. 2001
    (coastal altimetry book p. 458, EQ 17.1).

    Args:
        Hs:  significant wave height
        h:   satellite height over ground (m)
        tau: pulse duration (s)
    """
    c = 299792458  # m/s, speed of light
    R = 6371 * 10**3  # m, radius Earth
    r = np.sqrt(((c * tau + 2 * Hs) * h) / (1 + (h / R)))
    return r


def convert_meteorologic_oceanographic(alpha):
    """
    Convert angles between meteorological and oceanographic convention
    (and vice versa).
    """
    return (alpha + 180) % 360


def find_direction_convention(filevarname, ncdict):
    file_stdvarname = get_item_child(ncdict, filevarname)[0]["standard_name"]
    return file_stdvarname
