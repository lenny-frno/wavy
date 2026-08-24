#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------#
"""
Module to compute distance-to-ice-edge for observation/collocation
points, based on sea ice concentration (SIC) fields fetched from a
wavy model_class object.

Workflow:
    1. get_ice_field()            -> fetch SIC + lon/lat for one
                                      model timestep, via model_class
    2. extract_ice_edge_points()  -> contour the SIC field at a
                                      threshold to get ice-edge
                                      lon/lat points
    3. compute_distance_to_ice_edge() -> nearest-neighbour great
                                      circle distance from arbitrary
                                      points to the ice edge

These are combined per-unique-model-time by
collocation_class.add_dist_to_ice_edge() (see collocation_module.py).

Requires the model config (model_cfg.yaml) to define a variable
mapping for the requested ice varalias, e.g. for ww3_4km:

    vardef:
        SIC: ice
"""

# --- import libraries ------------------------------------------------#
import numpy as np
import logging
import pyresample as pr


from wavy.model_module import model_class as mc
from wavy.wconfig import load_or_default

# ---------------------------------------------------------------------#

variable_def = load_or_default("variable_def.yaml")

EARTH_RADIUS_M = 6371000.0


def get_ice_field(model_time, nID, leadtime=None, name=None, varalias="SIC", **kwargs):
    """
    Fetch the sea ice concentration field for a single model
    timestep, using the standard wavy model_class machinery.

    Args:
        model_time (datetime): the model timestep to fetch
        nID (str): model nID as defined in model_cfg.yaml
        leadtime (int|str|None): leadtime in hours, or 'best'/None
        name (str|None): optional model name override
        varalias (str): variable alias for ice concentration,
                        must be resolvable via the model's
                        vardef in model_cfg.yaml (default 'SIC')

    Returns:
        ice_lons, ice_lats, sic (np.ndarray, 2D each) or
        (None, None, None) if the field could not be retrieved.
    """
    logger = logging.getLogger(__name__)
    log_level = str(kwargs.get("logging", "WARNING").upper())
    logger.setLevel(getattr(logging, log_level, logging.WARNING))

    kwargs_clean = {
        k: v
        for k, v in kwargs.items()
        if k not in ("varalias", "sd", "ed", "nID", "leadtime", "name")
    }

    mco = mc(
        sd=model_time,
        ed=model_time,
        nID=nID,
        name=name,
        leadtime=leadtime,
        varalias=varalias,
        **kwargs_clean,
    ).populate(**kwargs_clean)

    if getattr(mco, "vars", None) is None:
        logger.warning(
            "Could not retrieve ice field ("
            + str(varalias)
            + ") for "
            + str(model_time)
            + " from model "
            + str(nID)
        )
        raise ValueError(
            "Could not retrieve ice field ("
            + str(varalias)
            + ") for "
            + str(model_time)
            + " from model "
            + str(nID)
        )
        # return None, None, None  # unreachable due to the exception

    try:
        ice_lons = np.array(mco.vars["lons"].values)
        ice_lats = np.array(mco.vars["lats"].values)
        sic = np.array(mco.vars[varalias].values)
    except KeyError as e:
        logger.warning("Ice variable missing in retrieved model data")
        logger.warning(e)
        return None, None, None

    # squeeze leading time dim if present, mirroring the pattern
    # used in collocation_class._collocate_field for model fields
    if sic.ndim > 2:
        sic = sic[0, ...].squeeze()
    if ice_lons.ndim > 2:
        ice_lons = ice_lons[0, ...].squeeze()
        ice_lats = ice_lats[0, ...].squeeze()
    elif ice_lons.ndim == 1:
        # regular lon/lat grid -> build 2D mesh to match sic shape
        ice_lons, ice_lats = np.meshgrid(ice_lons, ice_lats)

    if sic.shape != ice_lons.shape:
        logger.warning(
            "Shape mismatch between ice concentration field "
            + str(sic.shape)
            + " and coordinates "
            + str(ice_lons.shape)
        )
        return None, None, None

    return ice_lons, ice_lats, sic


def compute_distance_to_ice(
    pts_lons,
    pts_lats,
    ice_lons,
    ice_lats,
    sic,
    threshold=0.5,
):
    """
    Compute distance from each point to the nearest model grid point
    where SIC >= threshold.
    """

    pts_lons = np.asarray(pts_lons)
    pts_lats = np.asarray(pts_lats)

    # Select all model grid points above the SIC threshold
    mask = np.isfinite(sic) & (sic >= threshold)

    if not np.any(mask):
        return np.full(len(pts_lons), np.nan)

    # Coordinates of qualifying ice points
    ice_lons_points = ice_lons[mask]
    ice_lats_points = ice_lats[mask]

    # Observation points
    points_sdef = pr.geometry.SwathDefinition(
        lons=pts_lons,
        lats=pts_lats,
    )

    # Ice points
    ice_sdef = pr.geometry.SwathDefinition(
        lons=ice_lons_points,
        lats=ice_lats_points,
    )

    # Nearest ice point for every observation
    _, _, _, distance_array = pr.kd_tree.get_neighbour_info(
        ice_sdef,
        points_sdef,
        10000000,
        neighbours=1,
    )

    return distance_array


def get_dist_to_ice_edge(
    pts_lons,
    pts_lats,
    model_time,
    nID,
    leadtime=None,
    name=None,
    varalias="SIC",
    threshold=0.5,
    min_feature_cells=None,
    fill_enclosed_water=False,
    **kwargs,
):
    """
    Convenience wrapper combining the three steps above: fetch the
    ice field for one model timestep, extract the ice edge, and
    compute distances for a set of points at that timestep.

    See extract_ice_edge_points() for the meaning of
    min_feature_cells and fill_enclosed_water, which control how
    polynyas and isolated ice floes are handled.

    Returns:
        distances (np.ndarray), same length as pts_lons/pts_lats.
        All np.nan if the ice field could not be retrieved.
    """
    ice_lons, ice_lats, sic = get_ice_field(
        model_time, nID, leadtime=leadtime, name=name, varalias=varalias, **kwargs
    )

    if sic is None:
        return np.full(len(pts_lons), np.nan)
 
    return compute_distance_to_ice(pts_lons, pts_lats, ice_lons, ice_lats, sic, threshold=threshold)
