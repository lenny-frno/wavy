#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General-purpose utility functions (flatten, dict traversal, obs lookups).
"""

import numpy as np

from wavy.logmod import get_logger
from wavy.wconfig import load_or_default

logger = get_logger(__name__)

# flatten all lists before returning them
flatten = lambda l: [item for sublist in l for item in sublist]  # noqa: E731


def finditem(search_dict, field):
    """
    Takes a dict with nested lists and dicts, and searches all dicts for a
    key of the field provided.
    """
    fields_found = []
    for key, value in search_dict.items():
        if key == field:
            fields_found.append(value)
        elif isinstance(value, dict):
            results = finditem(value, field)
            for result in results:
                fields_found.append(result)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    more_results = finditem(item, field)
                    for another_result in more_results:
                        fields_found.append(another_result)
    return fields_found


def get_item_parent(ncdict, item, attr):
    lst = [
        i
        for i in ncdict.keys()
        if (attr in ncdict[i].keys() and item in ncdict[i][attr])
    ]
    if len(lst) >= 1:
        return lst
    else:
        return None


def get_item_child(ncdict, item):
    parent = finditem(ncdict, item)
    return parent


def get_obsdict(obstype):
    if obstype == "insitu":
        obsdict = load_or_default("insitu_specs.yaml")
    elif obstype == "satellite_altimeter":
        obsdict = load_or_default("satellite_specs.yaml")
    else:
        logger.warning("obstype '%s' is not applicable", obstype)
        obsdict = None
    return obsdict


def find_tagged_obs(tags, obstype):
    d = get_obsdict(obstype)
    l = []
    for t in tags:
        l += [k for k in d if t in d[k].get("tags", [""])]
    return list(np.unique(l))


def expand_nID_for_sensors(nID, obstype):
    obsdict = get_obsdict(obstype)
    sensors = list(obsdict[nID]["sensor"])
    return sensors
