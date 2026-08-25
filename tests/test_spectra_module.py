import xarray as xr
import numpy as np
import pytest

from wavy import spectra_module


@pytest.fixture(autouse=True)
def clear_spectral_file_cache():
    spectra_module.read_spectral_file.cache_clear()
    yield
    spectra_module.read_spectral_file.cache_clear()


def test_read_spectral_file_uses_ww3_backend_without_rescaling(monkeypatch):
    expected = xr.Dataset(
        {
            "efth": (("time", "site", "freq", "dir"), [[[[1.0]]]]),
            "wnd": (("time", "site"), [[11.0]]),
            "wnddir": (("time", "site"), [[225.0]]),
        },
        coords={
            "time": [0],
            "site": [0],
            "freq": [0.1],
            "dir": [45.0],
        },
    )
    called = {}

    def open_dataset(filename, **kwargs):
        called["filename"] = filename
        called.update(kwargs)
        return expected

    monkeypatch.setattr(spectra_module, "wavespectra", object())
    monkeypatch.setattr(xr, "open_dataset", open_dataset)

    actual = spectra_module.read_spectral_file("native_ww3.nc")

    assert called == {"filename": "native_ww3.nc", "engine": "ww3"}
    assert actual["efth"].item() == 1.0
    assert actual["wspd"].item() == 11.0
    assert actual["wdir"].item() == 225.0


def test_read_spectral_file_reports_ww3_parse_errors(monkeypatch):
    def open_dataset(*args, **kwargs):
        raise RuntimeError("not a native WW3 file")

    monkeypatch.setattr(spectra_module, "wavespectra", object())
    monkeypatch.setattr(xr, "open_dataset", open_dataset)

    with pytest.raises(ValueError, match="WW3 backend") as error:
        spectra_module.read_spectral_file("invalid.nc")

    assert "does not fall back" in str(error.value)


def _build_minimal_spectral_dataset():
    return xr.Dataset(
        {
            "efth": (
                ("time", "site", "freq", "dir"),
                np.ones((2, 2, 2, 2), dtype=float),
            ),
            "wspd": (("time", "site"), np.full((2, 2), 10.0)),
            "wdir": (("time", "site"), np.full((2, 2), 180.0)),
            "dpt": (("time", "site"), np.full((2, 2), 100.0)),
            "lon": (("site",), np.array([0.0, 10.0])),
            "lat": (("site",), np.array([0.0, 10.0])),
        },
        coords={
            "time": np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"),
            "site": [0, 1],
            "freq": [0.1, 0.2],
            "dir": [45.0, 135.0],
        },
    )


def test_collocate_spectra_reuses_unique_point_time_pairs(monkeypatch):
    ds = _build_minimal_spectral_dataset()

    calls = []

    def fake_partition_spectrum(spectrum, method="ptm1", **kwargs):
        time_value = np.datetime64(spectrum["time"].values)
        point_value = int(spectrum["site"].values)
        calls.append((time_value, point_value))
        return {"time": time_value, "point": point_value}

    def fake_compute_wave_regime_diagnostics(partitioned, **kwargs):
        code = partitioned["point"] + int(partitioned["time"].astype("datetime64[D]").astype(int))
        return {
            "wave_regime": float(code),
            "wind_sea_fraction": float(code + 0.1),
            "n_wave_systems": float(code + 0.2),
            "hs_total": float(code + 0.3),
        }

    monkeypatch.setattr(spectra_module, "partition_spectrum", fake_partition_spectrum)
    monkeypatch.setattr(
        spectra_module,
        "compute_wave_regime_diagnostics",
        fake_compute_wave_regime_diagnostics,
    )

    lons = np.array([0.0, 0.0, 10.0, 10.0])
    lats = np.array([0.0, 0.0, 10.0, 10.0])
    times = np.array(
        [
            "2020-01-01",
            "2020-01-01",
            "2020-01-02",
            "2020-01-02",
        ],
        dtype="datetime64[ns]",
    )

    result = spectra_module.collocate_spectra(ds, lons=lons, lats=lats, times=times)

    # Only two unique (time, point) combinations should be processed.
    assert len(calls) == 2

    # Repeated observations mapping to the same pair must get identical outputs.
    assert result["wave_regime"][0] == result["wave_regime"][1]
    assert result["wave_regime"][2] == result["wave_regime"][3]


def test_collocate_spectra_respects_max_time_difference_without_partition(monkeypatch):
    ds = _build_minimal_spectral_dataset()

    called = {"count": 0}

    def fake_partition_spectrum(*args, **kwargs):
        called["count"] += 1
        return {}

    monkeypatch.setattr(spectra_module, "partition_spectrum", fake_partition_spectrum)

    result = spectra_module.collocate_spectra(
        ds,
        lons=np.array([0.0]),
        lats=np.array([0.0]),
        times=np.array(["2030-01-01"], dtype="datetime64[ns]"),
        max_time_difference=1.0,
    )

    assert called["count"] == 0
    assert np.isnan(result["wave_regime"][0])
    assert np.isnan(result["wind_sea_fraction"][0])
    assert np.isnan(result["n_wave_systems"][0])
    assert np.isnan(result["hs_total"][0])
    assert result["spectral_point_index"][0] >= 0
    assert np.isfinite(result["spectral_distance_m"][0])
    assert np.isfinite(result["spectral_time_difference_s"][0])