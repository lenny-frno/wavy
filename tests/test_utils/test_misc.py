"""Tests for wavy.utils.misc."""

import numpy as np
import pytest

from wavy.utils.misc import finditem, flatten, get_item_child, get_item_parent


class TestFlatten:
    def test_simple_nested(self):
        assert flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]

    def test_empty_inner_lists(self):
        assert flatten([[], [1], []]) == [1]

    def test_empty_outer(self):
        assert flatten([]) == []

    def test_single_element_lists(self):
        assert flatten([[1], [2], [3]]) == [1, 2, 3]

    def test_preserves_order(self):
        assert flatten([[3, 1], [4, 1, 5]]) == [3, 1, 4, 1, 5]


class TestFinditem:
    nested = {
        "a": 1,
        "b": {"c": 2, "d": {"e": 3}},
        "f": [{"g": 4}, {"g": 5}],
    }

    def test_top_level_key(self):
        assert finditem(self.nested, "a") == [1]

    def test_deeply_nested_key(self):
        assert finditem(self.nested, "e") == [3]

    def test_key_present_in_list_of_dicts(self):
        result = finditem(self.nested, "g")
        assert set(result) == {4, 5}

    def test_missing_key_returns_empty_list(self):
        assert finditem(self.nested, "z") == []

    def test_multiple_occurrences_at_same_level(self):
        d = {"x": {"k": 1}, "y": {"k": 2}}
        result = finditem(d, "k")
        assert set(result) == {1, 2}


class TestGetItemParent:
    ncdict = {
        "var1": {
            "standard_name": "sea_surface_wave_significant_height",
            "units": "m",
        },
        "var2": {"standard_name": "wind_speed", "units": "m s-1"},
    }

    def test_finds_matching_variable(self):
        result = get_item_parent(self.ncdict, "significant_height", "standard_name")
        assert "var1" in result

    def test_no_match_returns_none(self):
        result = get_item_parent(self.ncdict, "temperature", "standard_name")
        assert result is None

    def test_finds_by_units(self):
        result = get_item_parent(self.ncdict, "m s-1", "units")
        assert "var2" in result


class TestGetItemChild:
    ncdict = {
        "time": {"standard_name": "time", "units": "seconds since 1970-01-01"},
        "Hs": {
            "standard_name": "sea_surface_wave_significant_height",
            "units": "m",
        },
    }

    def test_finds_existing_key(self):
        result = get_item_child(self.ncdict, "time")
        assert len(result) == 1
        assert result[0]["standard_name"] == "time"

    def test_missing_key_returns_empty_list(self):
        assert get_item_child(self.ncdict, "nonexistent") == []

    def test_finds_hs_entry(self):
        result = get_item_child(self.ncdict, "Hs")
        assert result[0]["units"] == "m"
