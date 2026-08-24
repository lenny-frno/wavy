import pytest

from wavy.satellite_module import satellite_class as sc
from wavy.collocation_module import collocation_class as cc
from wavy.insitu_module import insitu_class as ic
from wavy.insitu_module import poi_class as pc

# include possibility for collocating different variable
# varalias = 'Hs', 'U', aso...


def test_sat_collocation_and_validation(test_data, tmpdir):
    sd = "2022-2-1 12"
    ed = "2022-2-1 12"
    name = "s3a"
    varalias = "Hs"
    twin = 30
    nID = "cmems_L3_NRT"
    model = "ww3_4km"
    # init satellite_object and check for polygon region
    sco = sc(sd=sd, ed=ed, nID=nID, name=name, varalias=varalias, twin=twin)
    # read data
    sco = sco.populate(reader="read_local_ncfiles", path=str(test_data / "L3/s3a"))
    # crop to region
    sco = sco.crop_to_region(model)

    # collocate
    cco = cc(oco=sco, model=model, leadtime="best", distlim=6).populate()
    assert len(vars(cco).keys()) == 22
    assert len(cco.vars.keys()) == 10

    # validate


def test_cco_multivar(test_data):
    sd = "2022-2-1 12"
    ed = "2022-2-1 12"
    name = "s3a"
    varalias = "Hs"
    twin = 30
    nID = "cmems_L3_NRT"
    model = "ww3_4km"
    # init satellite_object and check for polygon region
    sco = sc(sd=sd, ed=ed, nID=nID, name=name, varalias=varalias, twin=twin)
    # read data
    sco = sco.populate(reader="read_local_ncfiles", path=str(test_data / "L3/s3a"))
    # crop to region
    sco = sco.crop_to_region(model)

    # collocate
    cco = cc(
        oco=sco, model=model, leadtime="best", distlim=6, varalias=["Hs", "Tm01"]
    ).populate()
    assert len(vars(cco).keys()) == 22
    assert len(cco.vars.keys()) == 11


def test_insitu_collocation_and_validation(test_data, tmpdir):
    sd = "2022-2-1 12"
    ed = "2022-2-1 12"
    varalias = "Hs"
    twin = 30
    model = "ww3_4km"
    nID = "D_Breisundet_wave"
    name = "wavescan"

    # init insitu_object and check for polygon region
    ico = ic(nID=nID, sd=sd, ed=ed, varalias=varalias, name=name, twin=twin)

    # read data
    ico = ico.populate()

    # collocate
    cco = cc(oco=ico, model=model, leadtime="best", distlim=6).populate()
    assert len(vars(cco).keys()) == 22
    assert len(cco.vars.keys()) == 10

    # validate


def test_insitu_collocation_leadtime(test_data, tmpdir):
    sd = "2024-01-01 10"
    ed = "2024-01-01 19"
    varalias = "Hs"
    twin = 30
    model = "ww3_4km"
    nID = "D_Breisundet_wave"
    name = "wavescan"

    # init insitu_object and check for polygon region
    ico = ic(nID=nID, sd=sd, ed=ed, varalias=varalias, name=name, twin=twin)

    # read data
    ico = ico.populate()

    # collocate
    cco = cc(oco=ico, model=model, leadtime=10, twin=9).populate()
    assert len(vars(cco).keys()) == 22
    assert len(cco.vars.keys()) == 10
    assert len(cco.vars.time) == 2


def test_poi_collocation():
    # define poi dictionary for track
    dt = ["2023-7-1", "2023-7-2", "2023-7-3"]
    lats = [56.5, 59.3, 64.3]
    lons = [3.5, 1.8, 4.2]
    poi_dict = {"time": dt, "lons": lons, "lats": lats}

    # init poi_class
    pco = pc(poi_dict)

    # collocate
    cco = cc(oco=pco, model="ww3_4km", leadtime="best").populate()
    assert len(vars(cco).keys()) == 22
    assert len(cco.vars.keys()) == 10


#    # write to nc
#    cco.write_to_nc(pathtofile=tmpdir.join('test.nc'))
#    # test validation
#    cco.validate_collocated_values()
#
# def test_insitu_collocation_and_validation():
#    sd = "2021-8-2 01"
#    ed = "2021-8-2 03"
#    nID = 'D_Breisundet_wave'
#    sensor = 'wavescan'
#    ico = ic(nID,sd,ed,varalias=varalias,stwin=1,date_incr=1,sensor=sensor)
#    # collocate
#    cco = cc(model='mwam4',obs_obj_in=ico,distlim=6,
#             leadtime='best',date_incr=1)
#    # test validation
#    cco.validate_collocated_values()


def test_collocate_observations(test_data):
    from wavy.collocation_module import collocate_observations

    sd = "2023-07-04"
    ed = "2023-07-05"
    ico = ic(sd=sd, ed=ed, nID="MO_Draugen_monthly", name="Draugen").populate(
        path=str(test_data / "insitu/monthly/Draugen/")
    )
    sco = sc(sd=sd, ed=ed, nID="cmems_L3_NRT", name="s3a").populate(
        path=str(test_data / "L3/s3a")
    )

    ico_colloc, sco_colloc = collocate_observations(ico, sco)
    print(ico_colloc)
    print(len(ico_colloc.vars.keys()))
    assert len(ico_colloc.vars.keys()) == 3
    assert len(ico_colloc.vars.time.values) > 0
    print(sco_colloc)
    print(len(sco_colloc.vars.keys()))
    assert len(sco_colloc.vars.keys()) == 4
    assert len(sco_colloc.vars.time.values) > 0

# def test_cco_add_dist_to_ice():
#     """
#     Checks that add_dist_to_ice appends a well-formed
#     'dist_to_ice' variable to an existing collocation_class
#     object, without altering the number of collocated points.
#     """
#     import numpy as np
 
#     # winter dates/high-latitude points, more likely to sit near
#     # or within the marginal ice zone than the poi_collocation
#     # test above, but the assertions below don't depend on ice
#     # actually being present at these particular points/dates
#     dt = ["2023-2-1", "2023-2-2", "2023-2-3"]
#     lats = [70.5, 74.0, 78.2]
#     lons = [20.0, 15.0, 10.0]
#     poi_dict = {"time": dt, "lons": lons, "lats": lats}
 
#     pco = pc(poi_dict)
 
#     cco = cc(oco=pco, model="ww3_4km", leadtime="best").populate()
#     n_points_before = len(cco.vars.time)
#     n_keys_before = len(cco.vars.keys())
 
#     cco = cco.add_dist_to_ice()
 
#     assert "dist_to_ice" in cco.vars.keys()
#     assert len(cco.vars.keys()) == n_keys_before + 1
#     # number of collocated points must not change
#     assert len(cco.vars.time) == n_points_before
 
#     dist = cco.vars["dist_to_ice"].values
#     assert len(dist) == n_points_before
#     # every entry is either NaN (no ice edge found for that
#     # timestep) or a finite, non-negative distance in meters
#     for d in dist:
#         assert np.isnan(d) or d >= 0.0
 
 
# def test_cco_add_dist_to_ice_options_do_not_change_length():
#     """
#     Checks that passing ice_threshold / ice-mask-cleanup kwargs
#     through add_dist_to_ice doesn't break the call chain and
#     still yields one value per collocated point.
#     """
#     import numpy as np
 
#     dt = ["2023-2-1", "2023-2-2"]
#     lats = [74.0, 78.2]
#     lons = [15.0, 10.0]
#     poi_dict = {"time": dt, "lons": lons, "lats": lats}
 
#     pco = pc(poi_dict)
#     cco = cc(oco=pco, model="ww3_4km", leadtime="best").populate()
#     n_points_before = len(cco.vars.time)
 
#     cco = cco.add_dist_to_ice(
#         ice_threshold=0.15,
#         min_feature_cells=5,
#         fill_enclosed_water=True,
#     )
 
#     assert "dist_to_ice" in cco.vars.keys()
#     dist = cco.vars["dist_to_ice"].values
#     assert len(dist) == n_points_before
#     for d in dist:
#         assert np.isnan(d) or d >= 0.0