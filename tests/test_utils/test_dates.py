"""Tests for wavy.utils.dates."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from wavy.utils.dates import (
    collocate_times,
    date_dispatcher,
    find_included_times,
    find_included_times_pd,
    hour_rounder,
    hour_rounder_pd,
    make_fc_dates,
    parse_date,
)


class TestParseDate:
    def test_passthrough_datetime(self):
        dt = datetime(2023, 6, 15, 12)
        assert parse_date(dt) is dt

    def test_parse_iso_string(self):
        assert parse_date("2023-06-15 12:00") == datetime(2023, 6, 15, 12, 0)

    def test_parse_compact_string(self):
        assert parse_date("20230615") == datetime(2023, 6, 15)

    def test_non_parseable_returns_as_is(self):
        # Non-string, non-datetime: should be returned unchanged
        assert parse_date(42) == 42


class TestHourRounder:
    def test_nearest_rounds_up(self):
        t = datetime(2023, 1, 1, 14, 35)
        assert hour_rounder(t) == datetime(2023, 1, 1, 15, 0)

    def test_nearest_rounds_down(self):
        t = datetime(2023, 1, 1, 14, 20)
        assert hour_rounder(t) == datetime(2023, 1, 1, 14, 0)

    def test_floor(self):
        t = datetime(2023, 1, 1, 14, 59)
        assert hour_rounder(t, method="floor") == datetime(2023, 1, 1, 14, 0)

    def test_ceil(self):
        t = datetime(2023, 1, 1, 14, 1)
        assert hour_rounder(t, method="ceil") == datetime(2023, 1, 1, 15, 0)

    def test_strips_seconds_and_microseconds(self):
        t = datetime(2023, 1, 1, 14, 10, 45, 999)
        result = hour_rounder(t)
        assert result.second == 0
        assert result.microsecond == 0


class TestHourRounderPd:
    def test_rounds_list_of_times(self):
        times = [datetime(2023, 1, 1, 14, 35), datetime(2023, 1, 1, 9, 10)]
        result = hour_rounder_pd(times)
        assert pd.Timestamp(result[0]) == pd.Timestamp("2023-01-01 15:00")
        assert pd.Timestamp(result[1]) == pd.Timestamp("2023-01-01 09:00")


class TestDateDispatcher:
    base = datetime(2023, 6, 15, 0, 0)

    def test_hour(self):
        assert date_dispatcher(self.base, "h", 3) == datetime(2023, 6, 15, 3)

    def test_day(self):
        assert date_dispatcher(self.base, "d", 2) == datetime(2023, 6, 17)

    def test_month(self):
        assert date_dispatcher(self.base, "m", 1) == datetime(2023, 7, 1)

    def test_year(self):
        assert date_dispatcher(self.base, "y", 1) == datetime(2024, 6, 1)

    def test_month_wraps_year(self):
        assert date_dispatcher(datetime(2023, 12, 1), "m", 1) == datetime(2024, 1, 1)


class TestMakeFcDates:
    def test_daily_vector(self):
        sd, ed = datetime(2023, 1, 1), datetime(2023, 1, 3)
        assert make_fc_dates(sd, ed, "d", 1) == [
            datetime(2023, 1, 1),
            datetime(2023, 1, 2),
            datetime(2023, 1, 3),
        ]

    def test_single_date(self):
        sd = datetime(2023, 6, 15)
        assert make_fc_dates(sd, sd, "d", 1) == [datetime(2023, 6, 15)]

    def test_string_inputs(self):
        assert len(make_fc_dates("2023-01-01", "2023-01-02", "d", 1)) == 2

    def test_hourly_vector(self):
        sd = datetime(2023, 1, 1, 0)
        ed = datetime(2023, 1, 1, 3)
        assert len(make_fc_dates(sd, ed, "h", 1)) == 4


class TestFindIncludedTimes:
    times = [datetime(2023, 1, 1, h) for h in range(6)]

    def test_twin_window(self):
        target = datetime(2023, 1, 1, 2, 30)
        idx = find_included_times(self.times, target_t=target, twin=60)
        assert 2 in idx and 3 in idx

    def test_twin_one_minute_includes_exact(self):
        # twin=0 gives empty window; use twin=1 to include the exact hour
        idx = find_included_times(self.times, target_t=datetime(2023, 1, 1, 2), twin=1)
        assert 2 in idx

    def test_sdate_edate_exclusive_upper(self):
        # upper bound is exclusive (strict <), so ed=03:00 excludes index 3
        sd = datetime(2023, 1, 1, 1)
        ed = datetime(2023, 1, 1, 3)
        assert find_included_times(self.times, sdate=sd, edate=ed) == [1, 2]

    def test_empty_result(self):
        idx = find_included_times(self.times, target_t=datetime(2023, 1, 2), twin=0)
        assert idx == []


class TestFindIncludedTimesPd:
    times = [datetime(2023, 1, 1, h) for h in range(6)]

    def test_range(self):
        sd = datetime(2023, 1, 1, 1)
        ed = datetime(2023, 1, 1, 4)
        idx = find_included_times_pd(self.times, sd, ed)
        assert 1 in idx and 2 in idx and 3 in idx
        assert 0 not in idx and 4 not in idx


class TestCollocateTimes:
    times = [datetime(2023, 1, 1, h) for h in range(6)]

    def test_exact_match_no_twin(self):
        target = [datetime(2023, 1, 1, 2), datetime(2023, 1, 1, 4)]
        idx = collocate_times(self.times, target_t=target, twin=0)
        assert set(idx) == {2, 4}

    def test_with_twin(self):
        target = [datetime(2023, 1, 1, 1, 30)]
        idx = collocate_times(self.times, target_t=target, twin=60)
        assert 1 in idx and 2 in idx

    def test_sdate_edate_exclusive_upper(self):
        # upper bound is exclusive (find_included_times uses strict <)
        sd = datetime(2023, 1, 1, 2)
        ed = datetime(2023, 1, 1, 3)
        idx = collocate_times(self.times, sdate=sd, edate=ed)
        assert 2 in idx
        assert 3 not in idx
