#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

from wavy.utils import make_pathtofile, make_subdict, parse_date
from wavy.wconfig import load_or_default
from wavy.spectra_module import read_spectral_file


model_dict = load_or_default("model_cfg.yaml")


def looks_like_url(path):
    return isinstance(path, str) and "://" in path


def has_glob_pattern(path):
    if not isinstance(path, str):
        return False
    return any(token in path for token in ["*", "?", "["])


def resolve_spectral_template(col_obj, **kwargs):
    """
    Resolve the spectral file template from kwargs or model_cfg.

    Supported options (in decreasing priority):
      - kwargs: spectral_path_tmplt
      - kwargs: spectral_src_tmplt + spectral_fl_tmplt
      - model_cfg[model]['spectral_input'] entries:
          path_tmplt OR src_tmplt + fl_tmplt
      - model_cfg[model]['wavy_input'] entries:
          spectral_path_tmplt OR spectral_src_tmplt + spectral_fl_tmplt
    """
    model_cfg = model_dict.get(col_obj.model, {})
    spectral_cfg = model_cfg.get("spectral_input", {})
    wavy_input = model_cfg.get("wavy_input", {})

    path_tmplt = kwargs.get("spectral_path_tmplt")
    src_tmplt = kwargs.get("spectral_src_tmplt")
    fl_tmplt = kwargs.get("spectral_fl_tmplt")

    if path_tmplt is None:
        path_tmplt = spectral_cfg.get("path_tmplt")
    if src_tmplt is None:
        src_tmplt = spectral_cfg.get("src_tmplt")
    if fl_tmplt is None:
        fl_tmplt = spectral_cfg.get("fl_tmplt")

    if path_tmplt is None:
        path_tmplt = wavy_input.get("spectral_path_tmplt")
    if src_tmplt is None:
        src_tmplt = wavy_input.get("spectral_src_tmplt")
    if fl_tmplt is None:
        fl_tmplt = wavy_input.get("spectral_fl_tmplt")

    if path_tmplt is None and src_tmplt is not None and fl_tmplt is not None:
        path_tmplt = src_tmplt + fl_tmplt

    if path_tmplt is None:
        return None

    strsub = (
        kwargs.get("spectral_strsub")
        or spectral_cfg.get("strsub")
        or wavy_input.get("strsub")
    )
    if strsub:
        subdict = make_subdict(strsub, class_object_dict=vars(col_obj))
        path_tmplt = make_pathtofile(path_tmplt, strsub, subdict, **kwargs)

    return path_tmplt


def resolve_spectral_files(col_obj, spectral_file=None, **kwargs):
    """
    Resolve one or more spectral files for wave-regime diagnostics.

    If spectral_file is None, files are built from model_cfg templates
    for each unique model timestamp. This naturally supports split files,
    e.g. one spectral file per month.
    """
    if spectral_file is None:
        path_tmplt = resolve_spectral_template(col_obj, **kwargs)
        if path_tmplt is None:
            raise ValueError(
                "spectral_file was not provided and no spectral template was "
                "found in model_cfg. Configure spectral_input.path_tmplt or "
                "spectral_input.src_tmplt + spectral_input.fl_tmplt for the "
                "selected model."
            )

        if hasattr(col_obj, "vars") and "model_time" in col_obj.vars:
            model_times = pd.to_datetime(col_obj.vars["model_time"].values)
            unique_times = pd.unique(model_times)
            dates = [pd.Timestamp(t).to_pydatetime() for t in unique_times]
        else:
            dates = [parse_date(str(col_obj.sd)), parse_date(str(col_obj.ed))]

        candidate_entries = [d.strftime(path_tmplt) for d in dates]
    elif isinstance(spectral_file, (list, tuple, set)):
        candidate_entries = list(spectral_file)
    else:
        candidate_entries = [spectral_file]

    files = []
    for entry in candidate_entries:
        if entry is None:
            continue

        if has_glob_pattern(entry):
            expanded = sorted(glob.glob(entry))
            files.extend(expanded)
            continue

        if looks_like_url(entry):
            files.append(entry)
            continue

        if os.path.exists(entry):
            files.append(entry)

    files = list(dict.fromkeys(files))

    if len(files) == 0:
        raise FileNotFoundError(
            "Could not resolve any spectral file. "
            "Provide spectral_file explicitly (path, glob pattern, or list), "
            "or update model_cfg spectral_input templates to match existing files."
        )

    return files


def read_and_merge_spectral_files(spectral_files, **kwargs):
    spectral_read_kwargs = dict(kwargs.get("spectral_read_kwargs", {}))
    if "spectral_chunks" in kwargs and "chunks" not in spectral_read_kwargs:
        spectral_read_kwargs["chunks"] = kwargs["spectral_chunks"]

    datasets = [read_spectral_file(path, **spectral_read_kwargs) for path in spectral_files]

    if len(datasets) == 1:
        return datasets[0]

    ds = xr.concat(
        datasets,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="outer",
    )

    if "time" in ds.coords:
        ds = ds.sortby("time")
        _, keep_idx = np.unique(ds["time"].values, return_index=True)
        if len(keep_idx) < ds.sizes.get("time", len(keep_idx)):
            ds = ds.isel(time=np.sort(keep_idx))

    return ds
