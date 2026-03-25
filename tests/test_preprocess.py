from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

from batteryops.data.preprocess import preprocess_dataset, resolve_loader


def test_rr_preprocess_outputs(tmp_path: Path) -> None:
    input_zip = tmp_path / "nasa_rr_battery.zip"
    output_dir = tmp_path / "processed"
    _write_rr_fixture(input_zip)

    outputs = preprocess_dataset(input_path=input_zip, output_dir=output_dir)

    samples = pd.read_parquet(outputs["telemetry_samples"])
    cycles = pd.read_parquet(outputs["cycle_features"])
    incidents = pd.read_parquet(outputs["incident_windows"])

    assert resolve_loader(input_zip).dataset_code == "nasa_rr"
    assert set(outputs.keys()) == {
        "telemetry_samples",
        "cycle_features",
        "incident_windows",
        "preprocess_manifest",
    }
    assert set(samples["asset_id"]) == {"battery00"}
    assert len(cycles) == 2
    assert len(incidents) == 1
    assert "low_voltage" in incidents["incident_types"].iloc[0]


def test_rw1_preprocess_outputs(tmp_path: Path) -> None:
    input_zip = tmp_path / "nasa_rw1_battery.zip"
    output_dir = tmp_path / "processed"
    _write_rw1_fixture(input_zip)

    outputs = preprocess_dataset(input_path=input_zip, output_dir=output_dir)

    samples = pd.read_parquet(outputs["telemetry_samples"])
    cycles = pd.read_parquet(outputs["cycle_features"])
    incidents = pd.read_parquet(outputs["incident_windows"])

    assert resolve_loader(input_zip).dataset_code == "nasa_rw1"
    assert set(samples["asset_id"]) == {"RW9"}
    assert set(cycles["cycle_kind"]) == {"step"}
    assert len(cycles) == 2
    assert len(incidents) == 1
    assert "high_temperature" in incidents["incident_types"].iloc[0]


def _write_rr_fixture(path: Path) -> None:
    header = ",".join(
        [
            "start_time",
            "time",
            "mode",
            "voltage_charger",
            "temperature_battery",
            "voltage_load",
            "current_load",
            "temperature_mosfet",
            "temperature_resistor",
            "mission_type",
        ]
    )
    csv_text = f"""{header}
2022-07-19 11:10:00,0,0,8.34,23.0,,,,,
2022-07-19 11:10:00,60,-1,8.10,25.0,3.90,2.50,31.0,29.0,0
2022-07-19 11:10:00,120,-1,7.95,41.0,3.15,4.20,42.0,40.0,0
2022-07-20 11:10:00,86400,1,8.25,NAN,,,,1
2022-07-20 11:10:00,86460,0,8.20,24.5,,,,1
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "battery_alt_dataset/regular_alt_batteries/battery00.csv",
            csv_text,
        )
        archive.writestr("battery_alt_dataset/README.txt", "fixture")


def _write_rw1_fixture(path: Path) -> None:
    mat_path = path.parent / "RW9.mat"
    steps = np.empty(2, dtype=object)
    steps[0] = {
        "comment": "reference discharge",
        "type": "D",
        "relativeTime": np.array([0.0, 60.0, 120.0]),
        "time": np.array([0.0, 60.0, 120.0]),
        "voltage": np.array([4.2, 3.9, 3.1]),
        "current": np.array([2.0, 2.0, 2.0]),
        "temperature": np.array([24.0, 25.0, 41.0]),
        "date": "06-Jan-2014 13:36:43",
    }
    steps[1] = {
        "comment": "rest post reference discharge",
        "type": "R",
        "relativeTime": np.array([0.0, 60.0]),
        "time": np.array([180.0, 240.0]),
        "voltage": np.array([3.8, 3.85]),
        "current": np.array([0.0, 0.0]),
        "temperature": np.array([24.0, 24.0]),
        "date": "06-Jan-2014 13:39:43",
    }
    savemat(
        mat_path,
        {"data": {"procedure": "RW fixture", "description": "synthetic", "step": steps}},
    )

    with zipfile.ZipFile(path, "w") as archive:
        archive.write(
            mat_path,
            arcname=(
                "Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/"
                "data/Matlab/RW9.mat"
            ),
        )
        archive.writestr(
            "Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/"
            "README_RW_ChargeDischarge_RT.Rmd",
            "fixture",
        )
