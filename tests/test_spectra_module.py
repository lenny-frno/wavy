import xarray as xr
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