#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date and time utility functions.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse

from wavy.logmod import get_logger
from wavy.utils.misc import flatten

logger = get_logger(__name__)


def parse_date(indate):
    """Parse a date string or datetime into a datetime object."""
    if isinstance(indate, datetime):
        return indate
    elif isinstance(indate, str):
        return parse(indate)
    else:
        logger.warning("Not able to parse input, returning as is")
        return indate


def hour_rounder(t, method="nearest"):
    """
    Rounds to nearest hour adding a timedelta hour if minute >= 30 (default),
    or to the integer hour before (floor) or the integer hour after (ceil) the
    given time.
    """
    if method == "nearest":
        add_hour = t.minute // 30
    elif method == "floor":
        add_hour = 0
    elif method == "ceil":
        add_hour = 1

    t = t.replace(second=0, microsecond=0, minute=0, hour=t.hour) + timedelta(
        hours=add_hour
    )
    return t


def hour_rounder_pd(times):
    """Rounds to nearest hour by adding a timedelta hour if minute >= 30."""
    df = pd.DataFrame(columns=["time"], data=times)
    rounded = df.time.dt.round("h").values
    return rounded


def date_dispatcher(date, date_incr="d", incr=1):
    dispatch_date = {
        "h": date_next_hour,
        "d": date_next_day,
        "m": date_next_month,
        "y": date_next_year,
    }
    return dispatch_date[date_incr](date, incr)


def date_next_hour(date, incr):
    date += timedelta(hours=incr)
    return date


def date_next_day(date, incr):
    date += timedelta(days=incr)
    return date


def date_next_month(date, incr):
    return datetime(
        (date + relativedelta(months=+incr)).year,
        (date + relativedelta(months=+incr)).month,
        1,
    )


def date_next_year(date, incr):
    return datetime(
        (date + relativedelta(years=+incr)).year,
        (date + relativedelta(years=+incr)).month,
        1,
    )


def make_fc_dates(
    sdate: datetime, edate: datetime, date_incr_unit: str, date_incr: int
) -> list:
    """Create a forecast date vector from sdate to edate."""
    sdate = parse_date(str(sdate))
    edate = parse_date(str(edate))
    fc_dates = []
    while sdate <= edate:
        fc_dates.append(sdate)
        tmp_date = parse_date(str(sdate))
        sdate = date_dispatcher(tmp_date, date_incr=date_incr_unit, incr=date_incr)
    return fc_dates


def find_included_times_pd(
    unfiltered_t: list, sdate: datetime, edate: datetime
) -> list:
    idx = np.array(range(len(unfiltered_t)))
    df = pd.to_datetime(unfiltered_t)
    mask = (df >= sdate.isoformat()) & (df < edate.isoformat())
    return list(idx[mask])


def find_included_times(
    unfiltered_t: list, target_t=None, sdate=None, edate=None, twin=0
) -> list:
    """
    Find index/indices of unfiltered time series that fall within a tolerance
    time window around the target time or within [sdate, edate].
    """
    if sdate is None and edate is None:
        idx = [
            i
            for i in range(len(unfiltered_t))
            if (
                unfiltered_t[i] >= target_t - timedelta(minutes=twin)
                and unfiltered_t[i] < target_t + timedelta(minutes=twin)
            )
        ]
    else:
        idx = [
            i
            for i in range(len(unfiltered_t))
            if (
                unfiltered_t[i] >= sdate - timedelta(minutes=twin)
                and unfiltered_t[i] < edate + timedelta(minutes=twin)
            )
        ]
    return idx


def collocate_times(
    unfiltered_t: list, target_t=None, sdate=None, edate=None, twin=None
) -> list:
    """
    Collocate times within a given twin tolerance.

    target_t and unfiltered_t must be lists of datetime objects.
    twin is in minutes.

    Returns indices.
    """
    if twin is None:
        twin = 0
    if (twin is None or twin == 0) and (sdate is None and edate is None):
        idx = [unfiltered_t.index(t) for t in target_t if t in unfiltered_t]
    else:
        if sdate is None and edate is None:
            idx = [
                find_included_times(
                    unfiltered_t, target_t=t, sdate=sdate, edate=edate, twin=twin
                )
                for t in target_t
            ]
            idx = flatten(idx)
        else:
            idx = find_included_times(unfiltered_t, sdate=sdate, edate=edate, twin=twin)
    return idx
