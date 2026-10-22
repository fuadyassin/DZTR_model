"""Regression checks for observation acquisition; no network requests are made."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import requests

from dztr_param import res_data_est_from_hydrometric as hydrometric


def _feature(value, *, date="2020-01-01", property_name="DISCHARGE"):
    return {"properties": {"DATE": date, property_name: value}}


def _response(features):
    return Mock(json=Mock(return_value={"features": features}))


@pytest.mark.parametrize("data_type", ["discharge", "level"])
def test_empty_response_remains_missing(monkeypatch, data_type):
    request = Mock(return_value=_response([]))
    monkeypatch.setattr(hydrometric.requests, "get", request)

    result = hydrometric.fetch_hydrometric_data_ca(["A"], "2020-01-01", "2020-01-02", data_type)

    assert result["Date"].tolist() == list(pd.date_range("2020-01-01", "2020-01-02"))
    assert result["value"].isna().all()
    assert request.call_args.kwargs["timeout"] == 30
    request.return_value.raise_for_status.assert_called_once_with()


def test_discharge_sum_requires_every_station(monkeypatch):
    request = Mock(side_effect=[
        _response([_feature(10), _feature(12, date="2020-01-02")]),
        _response([_feature(3, date="2020-01-02")]),
    ])
    monkeypatch.setattr(hydrometric.requests, "get", request)

    result = hydrometric.fetch_hydrometric_data_ca(["A", "B"], "2020-01-01", "2020-01-02")

    assert np.isnan(result.loc[0, "value"])
    assert result.loc[1, "value"] == 15


def test_station_with_no_response_prevents_partial_total(monkeypatch):
    request = Mock(side_effect=[_response([_feature(10)]), _response([])])
    monkeypatch.setattr(hydrometric.requests, "get", request)

    result = hydrometric.fetch_hydrometric_data_ca(["A", "B"], "2020-01-01", "2020-01-01")

    assert result["value"].isna().all()


@pytest.mark.parametrize("values", [[10, 10, None], [None, 10, 10]])
def test_duplicate_values_are_not_added_or_erased(monkeypatch, values):
    monkeypatch.setattr(hydrometric.requests, "get", Mock(return_value=_response([_feature(value) for value in values])))

    result = hydrometric.fetch_hydrometric_data_ca(["A"], "2020-01-01", "2020-01-01")

    assert result.loc[0, "value"] == 10


def test_duplicate_values_across_pages_are_not_added(monkeypatch):
    request = Mock(side_effect=[_response([_feature(10), _feature(10)]), _response([_feature(None)])])
    monkeypatch.setattr(hydrometric.requests, "get", request)

    result = hydrometric.fetch_hydrometric_data_ca(["A"], "2020-01-01", "2020-01-01", limit=2)

    assert result.loc[0, "value"] == 10
    assert request.call_args_list[1].kwargs["params"]["offset"] == 2


def test_conflicting_duplicate_values_raise(monkeypatch):
    monkeypatch.setattr(hydrometric.requests, "get", Mock(return_value=_response([_feature(10), _feature(11)])))

    with pytest.raises(ValueError, match="Conflicting discharge values for station A"):
        hydrometric.fetch_hydrometric_data_ca(["A"], "2020-01-01", "2020-01-01")


@pytest.mark.parametrize("stations", [[], [""], ["A", "A"], [" A ", "A"], "A", None])
def test_invalid_station_ids_fail_before_network(monkeypatch, stations):
    request = Mock()
    monkeypatch.setattr(hydrometric.requests, "get", request)

    with pytest.raises(ValueError):
        hydrometric.fetch_hydrometric_data_ca(stations, "2020-01-01", "2020-01-01")

    request.assert_not_called()


def test_multiple_level_stations_are_rejected(monkeypatch):
    request = Mock()
    monkeypatch.setattr(hydrometric.requests, "get", request)

    with pytest.raises(ValueError, match="exactly one station"):
        hydrometric.fetch_hydrometric_data_ca(["A", "B"], "2020-01-01", "2020-01-01", "level")

    request.assert_not_called()


def test_level_remains_one_station_measurement(monkeypatch):
    monkeypatch.setattr(hydrometric.requests, "get", Mock(return_value=_response([_feature(123.4, property_name="LEVEL")])))

    result = hydrometric.fetch_hydrometric_data_ca(["A"], "2020-01-01", "2020-01-02", "level")

    assert result.loc[0, "value"] == 123.4
    assert np.isnan(result.loc[1, "value"])


def test_http_errors_are_propagated(monkeypatch):
    response = _response([])
    response.raise_for_status.side_effect = requests.HTTPError("HTTP 500")
    monkeypatch.setattr(hydrometric.requests, "get", Mock(return_value=response))

    with pytest.raises(requests.HTTPError, match="HTTP 500"):
        hydrometric.fetch_hydrometric_data_ca(["A"], "2020-01-01", "2020-01-01")

    response.json.assert_not_called()


def test_geometry_first_point_and_numeric_sort_are_preserved(tmp_path):
    (tmp_path / "example.csv").write_text(
        "Dam Height (m)=10\nDepth(m),Area(skm),Storage(mcm)\n0,0,0\n10,100,200\n5,50,100\n",
        encoding="utf-8",
    )

    assert hydrometric.get_area_volume_from_csv("example", 10, tmp_path) == (0, 0)
    assert hydrometric.get_area_volume_from_csv("example", 7.5, tmp_path) == (25, 50)
