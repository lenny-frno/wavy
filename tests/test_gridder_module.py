import sys
import os
import numpy as np
from datetime import datetime
import pytest
from wavy import sc, gc
from wavy.grid_stats import apply_metric

def test_gridder_init(test_data):
    sd = "2022-2-1 12"
    ed = "2022-2-1 12"
    name = 's3a'
    varalias = ['Hs','U']
    twin = 30
    nID = 'cmems_L3_NRT'
    # init satellite_object
    sco = sc(sd=sd, ed=ed, nID=nID, name=name,
             varalias=varalias,
             twin=twin)
    # read data
    sco = sco.populate(reader='read_local_ncfiles',
                       path=str(test_data/"L3/s3a"))

    bb = (-179, 178, -80, 80)
    res = (5, 5) 
    gco = gc(oco=sco,bb=bb,res=res)
    assert len(vars(gco)) == 17   
    assert gco.varalias == 'Hs'
    assert gco.units == 'm'
    gridvar, lon_grid, lat_grid = apply_metric(gco=gco)
    assert len(gridvar.keys()) == 13


def test_assign_obs_to_grid_lower_bound_mapping():
    glons = np.array([0.0, 1.0, 2.0])
    glats = np.array([10.0, 11.0, 12.0])
    res = (1.0, 1.0)
    eps = 1e-12

    olons = np.array([0.0, -eps])
    olats = np.array([10.0, 10.0-eps])

    Midx = gc.assign_obs_to_grid(glons, glats, olons, olats, res)

    assert Midx[0][0] == 0
    assert Midx[1][0] == 0
    assert Midx[0][1] == -1
    assert Midx[1][1] == -1


def test_gridder_filters_points_just_below_lower_bound():
    eps = 1e-12
    gco = gc(
        lons=np.array([0.0, -eps]),
        lats=np.array([10.0, 10.0]),
        values=np.array([1.0, 2.0]),
        bb=(0.0, 2.0, 10.0, 12.0),
        res=(1.0, 1.0),
    )

    assert np.array_equal(gco.Midx_clean[0], np.array([0]))
    assert np.array_equal(gco.Midx_clean[1], np.array([0]))
    assert np.array_equal(gco.ovals_clean, np.array([1.0]))


@pytest.mark.parametrize(
    "date_input, expected",
    [
        (np.datetime64("2024-01-02T03:04:05"), "2024-01-02T03:04:05"),
        (datetime(2024, 1, 2, 3, 4, 5), "2024-01-02 03:04:05"),
        ("2024-01-02 03:04:05", "2024-01-02 03:04:05"),
        (np.array(np.datetime64("2024-01-02T03:04:05")), "2024-01-02T03:04:05"),
    ],
)
def test_format_title_date_mixed_inputs(date_input, expected):
    assert gc._format_title_date(date_input) == expected


def test_format_title_date_uses_data_attribute_when_present():
    class DummyXarrayLikeDate:
        def __init__(self, data):
            self.data = data

    date_input = DummyXarrayLikeDate(np.datetime64("2024-01-02T03:04:05"))
    assert gc._format_title_date(date_input) == "2024-01-02T03:04:05"




#def test_gridder_lowres(test_data, benchmark):
#    sco = sc(sdate="2020-11-1",edate="2020-11-3",region="global",
#             path_local=str(test_data/"L3"))
#    bb = (-170,170,-75,75)
#    res = (4, 4)
#    print("initialize")
#    gco = gc(oco=sco,bb=bb,res=res)
#    print("assign obs")
#    ovals,mvals,Midx = gco.get_obs_grid_idx()
#
#    ov, olo, ola = apply_metric(gco,metric="mean")
#    gv, glo, gla = apply_metric(gco,metric="mean_group")
#
#    # np.testing.assert_array_equal(olo, glo)
#    # np.testing.assert_array_equal(ola, gla)
#    np.testing.assert_array_almost_equal(ov, gv)
#
#    print("compute metric on grid")
#    var_gridded = benchmark(apply_metric, gco,metric="mean_group")
#
#def test_gridder_highres(test_data, benchmark):
#    sco = sc(sdate="2020-11-1",edate="2020-11-3",region="global",
#             path_local=str(test_data/"L3"))
#    bb = (-170,170,-75,75)
#    res = (.5, .5)
#    print("initialize")
#    gco = gc(oco=sco,bb=bb,res=res)
#    print("assign obs")
#    ovals,mvals,Midx = gco.get_obs_grid_idx()
#
#    ov, olo, ola = apply_metric(gco,metric="mean")
#    gv, glo, gla = apply_metric(gco,metric="mean_group")
#
#    # np.testing.assert_array_equal(olo, glo)
#    # np.testing.assert_array_equal(ola, gla)
#    np.testing.assert_array_almost_equal(ov, gv)
#
#    print("compute metric on grid")
#    var_gridded = benchmark(apply_metric, gco,metric="mean_group")
