"""
Backward-compatibility test: every public symbol must be importable directly
from wavy.utils (the package __init__.py re-exports everything).
"""

import pytest


def test_dates_symbols():
    from wavy.utils import (
        collocate_times,
        date_dispatcher,
        date_next_day,
        date_next_hour,
        date_next_month,
        date_next_year,
        find_included_times,
        find_included_times_pd,
        hour_rounder,
        hour_rounder_pd,
        make_fc_dates,
        parse_date,
    )
    for sym in [
        parse_date, hour_rounder, hour_rounder_pd, date_dispatcher,
        date_next_hour, date_next_day, date_next_month, date_next_year,
        make_fc_dates, find_included_times, find_included_times_pd,
        collocate_times,
    ]:
        assert callable(sym)


def test_geo_symbols():
    from wavy.utils import (
        convert_meteorologic_oceanographic,
        find_direction_convention,
        footprint_pulse_limited_radius,
        haversine_np,
        haversineA,
        haversineP,
    )
    for sym in [
        haversineP, haversine_np, haversineA, footprint_pulse_limited_radius,
        convert_meteorologic_oceanographic, find_direction_convention,
    ]:
        assert callable(sym)


def test_stats_symbols():
    from wavy.utils import (
        bootstr,
        calc_deep_water_T,
        calc_shallow_water_T,
        compute_quantiles,
        dispersion_deep_water,
        dispersion_intermediate_water,
        dispersion_shallow_water,
        marginalize,
        runmean,
        runmean_conv,
        runmean_old,
        wave_length_mask_swim,
    )
    for sym in [
        runmean_old, runmean, runmean_conv, bootstr, marginalize,
        compute_quantiles, dispersion_deep_water, dispersion_shallow_water,
        dispersion_intermediate_water, calc_deep_water_T, calc_shallow_water_T,
        wave_length_mask_swim,
    ]:
        assert callable(sym)


def test_io_symbols():
    from wavy.utils import (
        NoStdStreams,
        get_pathtofile,
        get_size,
        grab_PID,
        make_pathtofile,
        make_subdict,
        sort_aviso_l2p,
        sort_cci,
        sort_cmems_l3_my,
        sort_cmems_l3_nrt,
        sort_cmems_l3_s6a,
        sort_eumetsat_l2,
        sort_files,
        system_call,
    )
    assert callable(make_pathtofile)
    assert callable(NoStdStreams)


def test_misc_symbols():
    from wavy.utils import (
        expand_nID_for_sensors,
        find_tagged_obs,
        finditem,
        flatten,
        get_item_child,
        get_item_parent,
        get_obsdict,
    )
    for sym in [
        flatten, finditem, get_item_parent, get_item_child,
        get_obsdict, find_tagged_obs, expand_nID_for_sensors,
    ]:
        assert callable(sym)


def test_xr_tools_symbols():
    from wavy.utils import build_xr_ds, build_xr_ds_multivar

    assert callable(build_xr_ds)
    assert callable(build_xr_ds_multivar)
