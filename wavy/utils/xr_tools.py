#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xarray dataset builder utilities.
"""

import xarray as xr

from wavy.wconfig import load_or_default

variable_def = load_or_default("variable_def.yaml")


def build_xr_ds(var: tuple, varnames: tuple):
    ds = xr.Dataset(
        {
            varnames[0]: xr.DataArray(
                data=var[0],
                dims=[varnames[3]],
                coords={"time": var[3]},
                attrs=variable_def[varnames[0]],
            ),
            varnames[1]: xr.DataArray(
                data=var[1],
                dims=[varnames[3]],
                coords={"time": var[3]},
                attrs=variable_def[varnames[1]],
            ),
            varnames[2]: xr.DataArray(
                data=var[2],
                dims=[varnames[3]],
                coords={"time": var[3]},
                attrs=variable_def[varnames[2]],
            ),
            varnames[3]: xr.DataArray(
                data=var[3],
                dims=[varnames[3]],
                coords={"time": var[3]},
                attrs=variable_def[varnames[3]],
            ),
        },
        attrs={"title": "wavy dataset"},
    )
    return ds


def build_xr_ds_multivar(var: tuple, varnames: tuple, varalias: list):
    len_varalias = len(varalias)

    ds = xr.Dataset(
        {
            **{
                varnames[i]: xr.DataArray(
                    data=var[i],
                    dims=[varnames[len_varalias + 2]],
                    coords={"time": var[len_varalias + 2]},
                    attrs=variable_def[varnames[i]],
                )
                for i in range(len_varalias)
            },
            varnames[len_varalias]: xr.DataArray(
                data=var[len_varalias],
                dims=[varnames[len_varalias + 2]],
                coords={"time": var[len_varalias + 2]},
                attrs=variable_def[varnames[len_varalias]],
            ),
            varnames[len_varalias + 1]: xr.DataArray(
                data=var[len_varalias + 1],
                dims=[varnames[len_varalias + 2]],
                coords={"time": var[len_varalias + 2]},
                attrs=variable_def[varnames[len_varalias + 1]],
            ),
            varnames[len_varalias + 2]: xr.DataArray(
                data=var[len_varalias + 2],
                dims=[varnames[len_varalias + 2]],
                coords={"time": var[len_varalias + 2]},
                attrs=variable_def[varnames[len_varalias + 2]],
            ),
        },
        attrs={"title": "wavy dataset"},
    )
    return ds
