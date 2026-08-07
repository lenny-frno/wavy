#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File, path, and IO utility functions.
"""

import os
import sys
import subprocess
import netCDF4
from dateutil.parser import parse

from wavy.logmod import get_logger

logger = get_logger(__name__)


def grab_PID():
    """Retrieve and log the current process PID."""
    PID = os.getpid()
    logger.info("PID - with the license to kill :)  %s", PID)


def get_size(obj, seen=None):
    """
    Recursively find the size of an object.

    From: https://goshippo.com/blog/measure-real-size-any-python-object/
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def system_call(command: str):
    p = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True)
    return p.stdout.read()


class NoStdStreams(object):
    """
    Suppress stdout/stderr.

    Usage::

        with NoStdStreams():
            ...

    https://codereview.stackexchange.com/questions/25417/
    """

    def __init__(self, stdout=None, stderr=None):
        self.devnull = open(os.devnull, "w")
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()


def sort_files(dirpath, filelst, product, sat):
    """Move files to sub-folders of year and month."""
    if product == "cmems_L3_NRT":
        sort_cmems_l3_nrt(dirpath, filelst, sat)
    elif product == "cmems_L3_s6a":
        sort_cmems_l3_s6a(dirpath, filelst, sat)
    elif product == "cmems_L3_MY":
        sort_cmems_l3_my(dirpath, filelst, sat)
    elif product in ("cci_L2P", "cci_L3"):
        sort_cci(dirpath, filelst)
    elif product == "eumetsat_L2":
        sort_eumetsat_l2(dirpath, filelst)
    elif product == "cfo_swim_L2P":
        sort_aviso_l2p(dirpath, filelst)


def sort_aviso_l2p(dirpath: str, filelst: list):
    """Sort AVISO files according to year and month."""
    for e in filelst:
        if os.path.isfile(os.path.join(dirpath, e)):
            tmp = e.split("_")
            d1 = parse(tmp[-2])
            year, month = d1.strftime("%Y"), d1.strftime("%m")
            folder = os.path.join(dirpath, year, month)
            os.makedirs(folder, exist_ok=True)
            cmd = "mv " + dirpath + "/" + e + " " + folder
            os.system(cmd)


def sort_cmems_l3_nrt(dirpath: str, filelst: list, sat: str):
    """Sort L3 NRT files according to year and month."""
    for e in filelst:
        if os.path.isfile(os.path.join(dirpath, e)):
            tmp = "global_vavh_l3_rt_" + sat + "_"
            year, month = e[len(tmp) : len(tmp) + 4], e[len(tmp) + 4 : len(tmp) + 6]
            folder = os.path.join(dirpath, year, month)
            os.makedirs(folder, exist_ok=True)
            cmd = "mv " + dirpath + "/" + e + " " + folder
            os.system(cmd)


def sort_cmems_l3_s6a(dirpath: str, filelst: list, sat: str):
    """Sort L3 s6a files according to year and month."""
    for e in filelst:
        if os.path.isfile(os.path.join(dirpath, e)):
            tmp = "global_vavh_l3_rt_" + sat + "_lr_"
            year, month = e[len(tmp) : len(tmp) + 4], e[len(tmp) + 4 : len(tmp) + 6]
            folder = os.path.join(dirpath, year, month)
            os.makedirs(folder, exist_ok=True)
            cmd = "mv " + dirpath + "/" + e + " " + folder
            os.system(cmd)


def sort_cmems_l3_my(dirpath: str, filelst: list, sat: str):
    """Sort L3 MY files according to year and month."""
    for e in filelst:
        if os.path.isfile(os.path.join(dirpath, e)):
            tmp = "global_vavh_l3_rep_" + sat + "_"
            year, month = e[len(tmp) : len(tmp) + 4], e[len(tmp) + 4 : len(tmp) + 6]
            folder = os.path.join(dirpath, year, month)
            os.makedirs(folder, exist_ok=True)
            cmd = "mv " + dirpath + "/" + e + " " + folder
            os.system(cmd)


def sort_cci(dirpath: str, filelst: list):
    """Sort L2P and L3 CCI files according to year and month."""
    for e in filelst:
        if os.path.isfile(os.path.join(dirpath, e)):
            tmp = e.split("-")[-2]
            year, month = tmp[0:4], tmp[4:6]
            folder = os.path.join(dirpath, year, month)
            os.makedirs(folder, exist_ok=True)
            cmd = "mv " + dirpath + "/" + e + " " + folder
            os.system(cmd)


def sort_eumetsat_l2(dirpath: str, filelst: list):
    """Sort EUMETSAT L2 files according to year and month."""
    for e in filelst:
        splits = e.split("____")
        if os.path.isfile(os.path.join(dirpath, e)):
            year, month = splits[1][0:4], splits[1][4:6]
            folder = os.path.join(dirpath, year, month)
            logger.debug("Sorting %s -> %s/%s -> %s", e, year, month, folder)
            os.makedirs(folder, exist_ok=True)
            cmd = "mv " + dirpath + "/" + e + " " + folder
            os.system(cmd)


def make_subdict(strsublst, class_object=None, class_object_dict=None):
    """Build a substitution dict from a class object or dict."""
    if class_object_dict is None:
        class_object_dict = vars(class_object)
    subdict = {}
    if strsublst is None:
        pass
    else:
        for strsub in strsublst:
            if strsub in class_object_dict:
                subdict[strsub] = class_object_dict[strsub]
            else:
                logger.debug("%s is not available and not substituted", strsub)
    return subdict


def make_pathtofile(tmppath, strsublst, subdict, date=None, **kwargs):
    """Create a path given templates, keywords, and optional date."""
    if date is not None:
        pathtofile = date.strftime(tmppath)
    else:
        pathtofile = tmppath
    if strsublst is None:
        pass
    else:
        for strsub in strsublst:
            if strsub in subdict:
                pathtofile = pathtofile.replace(strsub, subdict[strsub])
            else:
                logger.debug(
                    "%s in substitutables not needed for destination path", strsub
                )
    return pathtofile


def get_pathtofile(pathlst, strsublst, subdict, date):
    """
    Find and return path of file given templates, keywords, and date.
    """
    i = 0
    switch = False
    if not isinstance(pathlst, list):
        pathlst = [pathlst]
    while switch is False:
        try:
            pathtofile = date.strftime(pathlst[i])
        except IndexError as e:
            logger.error("%s — index too large for pathlst, returning None", e)
            return None
        for strsub in strsublst:
            pathtofile = pathtofile.replace(strsub, subdict[strsub])
        # check if thredds and accessible
        if "thredds" in pathtofile and pathtofile[-3:] == ".nc":
            try:
                nc = netCDF4.Dataset(pathtofile)
                nc.close()
                switch = True
            except Exception as e:
                logger.warning("%s — %s not accessible", e, pathtofile)
        else:
            if os.path.isfile(pathtofile) is not False:
                switch = True
        if switch is False:
            logger.debug("%s does not exist, trying next", pathtofile)
            i += 1
    return pathtofile
