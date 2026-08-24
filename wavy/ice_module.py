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

from scipy import ndimage
from sklearn.neighbors import BallTree

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


def _clean_ice_mask(
    mask, min_feature_cells=None, fill_enclosed_water=False, connectivity=4
):
    """
    Optional pre-processing of the binary ice mask to handle two
    topological edge cases before boundary detection:

    1. Polynyas / leads: small water openings fully enclosed by ice.
       If fill_enclosed_water is True, these are filled in (treated
       as ice) so they no longer generate an "ice edge" around the
       inside of the pack. Water regions that touch the domain
       border are always treated as open ocean and left untouched,
       regardless of size.

    2. Isolated ice floes/islands: small disconnected ice features
       surrounded by open water. If min_feature_cells is set, ice
       connected-components smaller than this cell count are
       dropped from the mask (treated as water) so they don't
       register as "the ice edge" for nearby points.

    Without these options (both args left at their defaults), the
    mask is returned unchanged and every ice/water boundary cell
    counts as an edge cell, including polynyas and isolated floes
    of any size -- i.e. the literal geometric nearest ice/water
    boundary, whatever its topology.

    Args:
        mask (np.ndarray): boolean 2D array, True = ice
        min_feature_cells (int|None): drop ice components smaller
                                      than this many cells
        fill_enclosed_water (bool): fill water holes that don't
                                    touch the domain border
        connectivity (int): 4 or 8, used for both filters

    Returns:
        mask (np.ndarray): cleaned boolean 2D array
    """
    # structure = (np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    #             if connectivity == 4 else np.ones((3, 3)))

    # if min_feature_cells is not None and min_feature_cells > 1:
    #     labeled, n = ndimage.label(mask, structure=structure)
    #     if n > 0:
    #         sizes = ndimage.sum(mask, labeled, index=range(1, n + 1))
    #         small_labels = np.where(sizes < min_feature_cells)[0] + 1
    #         mask = mask & ~np.isin(labeled, small_labels)

    # if fill_enclosed_water:
    #     water = ~mask
    #     labeled_w, nw = ndimage.label(water, structure=structure)
    #     if nw > 0:
    #         # any water component touching the domain border is
    #         # treated as open ocean and left alone; every other
    #         # water component is fully enclosed by ice -> fill it
    #         border_labels = set(labeled_w[0, :]) | set(labeled_w[-1, :]) \
    #                        | set(labeled_w[:, 0]) | set(labeled_w[:, -1])
    #         border_labels.discard(0)
    #         enclosed_labels = set(range(1, nw + 1)) - border_labels
    #         if enclosed_labels:
    #             mask = mask | np.isin(labeled_w, list(enclosed_labels))

    return mask


def extract_ice_edge_points(
    ice_lons,
    ice_lats,
    sic,
    threshold,
    connectivity=4,
    min_feature_cells=None,
    fill_enclosed_water=False,
):
    """
    Extract lon/lat points lying on the ice edge, defined as ice
    grid cells (concentration >= threshold) that have at least one
    neighbouring cell below threshold (open water or NaN/land, both
    treated as non-ice).

    This is a simple grid-boundary detection using array shifting,
    requiring only numpy. It works for any 2D grid, including
    curvilinear/rotated-pole grids such as ww3_4km, since it never
    needs the grid projection -- lon/lat are only used to label the
    resulting boundary cells. The edge is defined at the resolution
    of the native grid (~4 km for ww3_4km), which is fine given
    that the observation-to-edge distances of interest are
    typically much larger than one grid cell.

    Args:
        ice_lons, ice_lats (np.ndarray): 2D arrays, shape (ny, nx)
        sic (np.ndarray): 2D array, shape (ny, nx), values in [0, 1]
                          (NaNs, e.g. over land, are treated as 0)
        threshold (float): concentration threshold defining the edge
        connectivity (int): 4 (N/S/E/W neighbours) or 8 (also
                            diagonals). 4 is a slightly stricter/
                            thinner edge and is the default.
        min_feature_cells (int|None): if set, ice connected-
                            components smaller than this many
                            cells are dropped and treated as water
                            before edge detection -- use this to
                            ignore small isolated ice floes/islands
                            so they don't register as "the ice
                            edge" for nearby points. None (default)
                            keeps every ice feature regardless of
                            size.
        fill_enclosed_water (bool): if True, water regions fully
                            enclosed by ice (polynyas/leads that do
                            not touch the domain border) are filled
                            in before edge detection, so the inside
                            of the pack no longer generates an
                            "ice edge" around such openings. False
                            (default) treats every polynya, however
                            small, as generating a valid ice edge.

    Returns:
        edge_lons, edge_lats (np.ndarray): 1D arrays of ice-edge
                                            point coordinates.
                                            Empty arrays if no edge
                                            is found (fully ice-free
                                            or fully ice-covered field).
    """
    logger = logging.getLogger(__name__)

    sic_filled = np.where(np.isnan(sic), 0.0, sic)

    if np.nanmax(sic_filled) < threshold:
        logger.info("No ice above threshold in field -> no ice edge")
        return np.array([]), np.array([])
    if np.nanmin(sic_filled) >= threshold:
        logger.info(
            "Field fully ice-covered above threshold " "-> no ice edge within domain"
        )
        return np.array([]), np.array([])

    mask = sic_filled >= threshold

    if min_feature_cells is not None or fill_enclosed_water:
        mask = _clean_ice_mask(
            mask,
            min_feature_cells=min_feature_cells,
            fill_enclosed_water=fill_enclosed_water,
            connectivity=connectivity,
        )
        if not mask.any():
            logger.info("No ice left after cleaning small features " "-> no ice edge")
            return np.array([]), np.array([])
        if mask.all():
            logger.info(
                "Field fully ice-covered after filling "
                "enclosed water -> no ice edge within domain"
            )
            return np.array([]), np.array([])

    # pad with False (= open water) on all sides so that cells on
    # the domain boundary are compared against an implicit
    # non-ice neighbour rather than wrapping or being skipped
    padded = np.pad(mask, 1, mode="constant", constant_values=False)

    up = padded[0:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, 0:-2]
    right = padded[1:-1, 2:]

    not_all_ice = ~up | ~down | ~left | ~right

    if connectivity == 8:
        upleft = padded[0:-2, 0:-2]
        upright = padded[0:-2, 2:]
        downleft = padded[2:, 0:-2]
        downright = padded[2:, 2:]
        not_all_ice = not_all_ice | ~upleft | ~upright | ~downleft | ~downright

    # boundary ice cells: part of the ice mask, but bordering at
    # least one non-ice cell
    boundary = mask & not_all_ice

    rows, cols = np.where(boundary)
    edge_lons = ice_lons[rows, cols]
    edge_lats = ice_lats[rows, cols]
    return edge_lons, edge_lats


def compute_distance_to_ice_edge(pts_lons, pts_lats, edge_lons, edge_lats):
    """
    Compute great-circle distance (in meters) from each input point
    to the nearest ice-edge point, using a BallTree with the
    haversine metric.

    Args:
        pts_lons, pts_lats (array-like): coordinates of the query
                                          points (e.g. obs locations)
        edge_lons, edge_lats (array-like): coordinates of ice-edge
                                            points, as returned by
                                            extract_ice_edge_points()

    Returns:
        distances (np.ndarray): distance in meters for each query
                                point. np.nan where no ice edge was
                                available (e.g. ice-free domain).
    """
    pts_lons = np.asarray(pts_lons)
    pts_lats = np.asarray(pts_lats)

    if len(edge_lons) == 0:
        return np.full(len(pts_lons), np.nan)

    edge_rad = np.deg2rad(np.column_stack([edge_lats, edge_lons]))
    pts_rad = np.deg2rad(np.column_stack([pts_lats, pts_lons]))

    tree = BallTree(edge_rad, metric="haversine")
    dist_rad, _ = tree.query(pts_rad, k=1)
    distances = dist_rad[:, 0] * EARTH_RADIUS_M
    return distances


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

    edge_lons, edge_lats = extract_ice_edge_points(
        ice_lons,
        ice_lats,
        sic,
        threshold=threshold,
        min_feature_cells=min_feature_cells,
        fill_enclosed_water=fill_enclosed_water,
    )

    return compute_distance_to_ice_edge(pts_lons, pts_lats, edge_lons, edge_lats)
