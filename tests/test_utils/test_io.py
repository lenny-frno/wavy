"""Tests for wavy.utils.io."""

import io
import sys
from datetime import datetime

import pytest

from wavy.utils.io import NoStdStreams, get_size, make_pathtofile, make_subdict


class TestMakeSubdict:
    def test_extracts_present_keys(self):
        result = make_subdict(["a", "b"], class_object_dict={"a": 1, "b": 2, "c": 3})
        assert result == {"a": 1, "b": 2}

    def test_missing_key_silently_excluded(self):
        result = make_subdict(["x"], class_object_dict={"a": 1})
        assert "x" not in result
        assert result == {}

    def test_none_strsublst_returns_empty(self):
        result = make_subdict(None, class_object_dict={"a": 1})
        assert result == {}

    def test_uses_class_object_when_dict_absent(self):
        class Dummy:
            pass

        obj = Dummy()
        obj.nID = "s3a"
        result = make_subdict(["nID"], class_object=obj)
        assert result == {"nID": "s3a"}


class TestMakePathtofile:
    def test_simple_substitution(self):
        path = make_pathtofile("/data/{nID}/file.nc", ["{nID}"], {"{nID}": "s3a"})
        assert path == "/data/s3a/file.nc"

    def test_date_strftime(self):
        date = datetime(2023, 6, 15)
        path = make_pathtofile("/data/%Y/%m/file.nc", None, {}, date=date)
        assert path == "/data/2023/06/file.nc"

    def test_missing_key_leaves_template_unchanged(self):
        path = make_pathtofile("/data/{nID}/file.nc", ["{nID}"], {})
        assert "{nID}" in path

    def test_no_strsublst_returns_path_as_is(self):
        path = make_pathtofile("/data/fixed/file.nc", None, {})
        assert path == "/data/fixed/file.nc"

    def test_multiple_substitutions(self):
        path = make_pathtofile(
            "/{nID}/{var}.nc",
            ["{nID}", "{var}"],
            {"{nID}": "s3a", "{var}": "Hs"},
        )
        assert path == "/s3a/Hs.nc"


class TestNoStdStreams:
    def test_redirects_stdout_to_provided_buffer(self):
        captured = io.StringIO()
        with NoStdStreams(stdout=captured):
            print("hello")
        assert captured.getvalue() == "hello\n"

    def test_restores_stdout_on_normal_exit(self):
        original = sys.stdout
        with NoStdStreams():
            pass
        assert sys.stdout is original

    def test_restores_stdout_on_exception(self):
        original = sys.stdout
        try:
            with NoStdStreams():
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert sys.stdout is original


class TestGetSize:
    def test_returns_positive_for_int(self):
        assert get_size(42) > 0

    def test_list_larger_than_empty_list(self):
        assert get_size([1, 2, 3]) > get_size([])

    def test_nested_dict_larger_than_empty(self):
        assert get_size({"a": {"b": 1}}) > get_size({})

    def test_string_size_grows_with_length(self):
        assert get_size("a" * 100) > get_size("a")
