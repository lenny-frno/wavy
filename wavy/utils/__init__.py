#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wavy.utils package — re-exports all public utilities for backward compatibility.

Internal layout:

  wavy.utils.dates     — date/time helpers
                         (parse_date, hour_rounder, make_fc_dates,
                          date_dispatcher, find_included_times, collocate_times …)

  wavy.utils.geo       — geospatial/distance helpers
                         (haversineA, footprint_pulse_limited_radius,
                          convert_meteorologic_oceanographic, find_direction_convention …)

  wavy.utils.stats     — statistical and wave-physics helpers
                         (runmean, bootstr, marginalize, compute_quantiles,
                          calc_deep_water_T, wave_length_mask_swim …)

  wavy.utils.io        — file/path/IO helpers
                         (make_pathtofile, get_pathtofile, make_subdict,
                          sort_files, NoStdStreams …)

  wavy.utils.misc      — general helpers
                         (flatten, finditem, get_item_parent, get_item_child,
                          get_obsdict, find_tagged_obs, expand_nID_for_sensors)

  wavy.utils.xr_tools  — xarray dataset builders
                         (build_xr_ds, build_xr_ds_multivar)

All names are re-exported here so that ``from wavy.utils import X`` continues to
work for existing code.
"""

from wavy.utils.dates import (
    parse_date,
    hour_rounder,
    hour_rounder_pd,
    make_fc_dates,
    date_dispatcher,
    date_next_hour,
    date_next_day,
    date_next_month,
    date_next_year,
    find_included_times_pd,
    find_included_times,
    collocate_times,
)
from wavy.utils.geo import (
    haversineP,
    haversine_np,
    haversineA,
    footprint_pulse_limited_radius,
    convert_meteorologic_oceanographic,
    find_direction_convention,
)
from wavy.utils.stats import (
    runmean_old,
    runmean,
    runmean_conv,
    bootstr,
    marginalize,
    compute_quantiles,
    dispersion_deep_water,
    dispersion_shallow_water,
    dispersion_intermediate_water,
    calc_deep_water_T,
    calc_shallow_water_T,
    wave_length_mask_swim,
)
from wavy.utils.io import (
    grab_PID,
    get_size,
    system_call,
    sort_files,
    sort_aviso_l2p,
    sort_cmems_l3_nrt,
    sort_cmems_l3_s6a,
    sort_cmems_l3_my,
    sort_cci,
    sort_eumetsat_l2,
    make_pathtofile,
    get_pathtofile,
    make_subdict,
    NoStdStreams,
)
from wavy.utils.misc import (
    flatten,
    finditem,
    get_item_parent,
    get_item_child,
    get_obsdict,
    find_tagged_obs,
    expand_nID_for_sensors,
)
from wavy.utils.xr_tools import (
    build_xr_ds,
    build_xr_ds_multivar,
)

__all__ = [
    # dates
    "parse_date",
    "hour_rounder",
    "hour_rounder_pd",
    "make_fc_dates",
    "date_dispatcher",
    "date_next_hour",
    "date_next_day",
    "date_next_month",
    "date_next_year",
    "find_included_times_pd",
    "find_included_times",
    "collocate_times",
    # geo
    "haversineP",
    "haversine_np",
    "haversineA",
    "footprint_pulse_limited_radius",
    "convert_meteorologic_oceanographic",
    "find_direction_convention",
    # stats
    "runmean_old",
    "runmean",
    "runmean_conv",
    "bootstr",
    "marginalize",
    "compute_quantiles",
    "dispersion_deep_water",
    "dispersion_shallow_water",
    "dispersion_intermediate_water",
    "calc_deep_water_T",
    "calc_shallow_water_T",
    "wave_length_mask_swim",
    # io
    "grab_PID",
    "get_size",
    "system_call",
    "sort_files",
    "sort_aviso_l2p",
    "sort_cmems_l3_nrt",
    "sort_cmems_l3_s6a",
    "sort_cmems_l3_my",
    "sort_cci",
    "sort_eumetsat_l2",
    "make_pathtofile",
    "get_pathtofile",
    "make_subdict",
    "NoStdStreams",
    # misc
    "flatten",
    "finditem",
    "get_item_parent",
    "get_item_child",
    "get_obsdict",
    "find_tagged_obs",
    "expand_nID_for_sensors",
    # xr_tools
    "build_xr_ds",
    "build_xr_ds_multivar",
]
