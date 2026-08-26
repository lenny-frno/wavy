from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from wavy import collocation_module as cm
from wavy import spectral_readers as sr


class DummyCollocation:
    def __init__(self, model, vars_ds):
        self.model = model
        self.vars = vars_ds
        self.sd = datetime(2023, 1, 1)
        self.ed = datetime(2023, 2, 28)
        self.leadtime = 0
        self.name = None


def _build_dummy_vars():
    return xr.Dataset(
        {
            "model_time": (
                ("time",),
                np.array(
                    [
                        "2023-01-15T00:00:00",
                        "2023-02-10T00:00:00",
                    ],
                    dtype="datetime64[ns]",
                ),
            ),
            "obs_lons": (("time",), np.array([3.0, 4.0])),
            "obs_lats": (("time",), np.array([60.0, 61.0])),
        },
        coords={"time": np.array([0, 1])},
    )


def test_resolve_spectral_files_from_model_cfg_monthly_split(tmp_path, monkeypatch):
    model_name = "test_model_spectral_split"
    model_dir = tmp_path / "storm_greenland_20230202"
    model_dir.mkdir(parents=True)

    jan = model_dir / "ww3.202301_spec.nc"
    feb = model_dir / "ww3.202302_spec.nc"
    jan.touch()
    feb.touch()

    cfg = {
        model_name: {
            "spectral_input": {
                "path_tmplt": str(model_dir / "ww3.%Y%m_spec.nc"),
            }
        }
    }
    monkeypatch.setattr(sr, "model_dict", cfg)

    obj = DummyCollocation(model=model_name, vars_ds=_build_dummy_vars())

    files = sr.resolve_spectral_files(obj, spectral_file=None)

    assert files == [str(jan), str(feb)]


def test_add_wave_regime_uses_configured_spectral_files_when_optional(monkeypatch, tmp_path):
    model_name = "test_model_add_wave_regime"
    model_dir = tmp_path / "storm"
    model_dir.mkdir(parents=True)

    jan = model_dir / "ww3.202301_spec.nc"
    feb = model_dir / "ww3.202302_spec.nc"
    jan.touch()
    feb.touch()

    cfg = {
        model_name: {
            "spectral_input": {
                "path_tmplt": str(model_dir / "ww3.%Y%m_spec.nc"),
            }
        }
    }
    monkeypatch.setattr(sr, "model_dict", cfg)

    read_calls = []

    def fake_read_spectral_file(path, **kwargs):
        read_calls.append(path)
        return xr.Dataset(
            {"efth": (("time", "site", "freq", "dir"), np.ones((1, 1, 1, 1)))},
            coords={
                "time": np.array(["2023-01-01"], dtype="datetime64[ns]"),
                "site": [0],
                "freq": [0.1],
                "dir": [90.0],
            },
        )

    def fake_collocate_spectra(ds, lons, lats, times, **kwargs):
        n = len(lons)
        return {
            "wave_regime": np.zeros(n),
            "wind_sea_fraction": np.full(n, 0.8),
            "n_wave_systems": np.ones(n),
            "hs_total": np.full(n, 2.5),
        }

    import wavy.spectra_module as spectra_module

    monkeypatch.setattr(sr, "read_spectral_file", fake_read_spectral_file)
    monkeypatch.setattr(spectra_module, "collocate_spectra", fake_collocate_spectra)

    obj = DummyCollocation(model=model_name, vars_ds=_build_dummy_vars())

    out = cm.collocation_class.add_wave_regime(obj, spectral_file=None)

    assert read_calls == [str(jan), str(feb)]
    assert "wave_regime" in out.vars
    assert "wind_sea_fraction" in out.vars
    assert "n_wave_systems" in out.vars
    assert "hs_total" in out.vars
    assert np.all(np.isfinite(out.vars["hs_total"].values))


def test_add_wave_regime_requires_file_or_config(monkeypatch):
    monkeypatch.setattr(sr, "model_dict", {})
    obj = DummyCollocation(model="unknown", vars_ds=_build_dummy_vars())

    with pytest.raises(ValueError, match="spectral_file was not provided"):
        cm.collocation_class.add_wave_regime(obj, spectral_file=None)
