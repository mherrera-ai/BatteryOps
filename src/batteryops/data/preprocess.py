from __future__ import annotations

import argparse
import io
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat  # type: ignore[import-untyped]

from batteryops.features import build_cycle_features, build_incident_windows

EXPECTED_INPUTS = ("nasa_rr_battery.zip", "nasa_rw1_battery.zip")
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_OUTPUT_DIR = Path("data/processed")


@dataclass(frozen=True)
class LoadedBatteryDataset:
    dataset_code: str
    dataset_name: str
    cycle_kind: str
    samples: pd.DataFrame


class BatteryDatasetLoader(ABC):
    dataset_code: str
    dataset_name: str
    cycle_kind: str
    accepted_filenames: tuple[str, ...]

    @abstractmethod
    def supports(self, zip_path: Path, members: list[str]) -> bool:
        """Return True when this loader can parse the given archive."""

    @abstractmethod
    def load(self, zip_path: Path, max_assets: int | None = None) -> LoadedBatteryDataset:
        """Load and normalize archive contents into sample-level telemetry."""


class RandomizedRecommissionedLoader(BatteryDatasetLoader):
    dataset_code = "nasa_rr"
    dataset_name = "Randomized and Recommissioned Battery Dataset"
    cycle_kind = "start_time_cycle"
    accepted_filenames = ("nasa_rr_battery.zip",)

    COLUMN_MAP = {
        "start_time": "cycle_start_time",
        "time": "experiment_time_s",
        "mode": "mode_code",
        "voltage_charger": "voltage_charger_v",
        "temperature_battery": "temperature_c",
        "voltage_load": "voltage_load_v",
        "current_load": "current_a",
        "temperature_mosfet": "temperature_mosfet_c",
        "temperature_resistor": "temperature_resistor_c",
        "mission_type": "mission_type_code",
    }
    NUMERIC_COLUMNS = (
        "experiment_time_s",
        "mode_code",
        "voltage_charger_v",
        "temperature_c",
        "voltage_load_v",
        "current_a",
        "temperature_mosfet_c",
        "temperature_resistor_c",
        "mission_type_code",
    )

    def supports(self, zip_path: Path, members: list[str]) -> bool:
        has_expected_name = zip_path.name in self.accepted_filenames
        has_csv_members = any(member.endswith(".csv") for member in members)
        has_root = any(member.startswith("battery_alt_dataset/") for member in members)
        return has_expected_name or (has_csv_members and has_root)

    def load(self, zip_path: Path, max_assets: int | None = None) -> LoadedBatteryDataset:
        frames: list[pd.DataFrame] = []
        with zipfile.ZipFile(zip_path) as archive:
            csv_members = [
                member
                for member in sorted(archive.namelist())
                if member.endswith(".csv") and not member.startswith("__MACOSX/")
            ]
            if max_assets is not None:
                csv_members = csv_members[:max_assets]

            for member in csv_members:
                with archive.open(member) as handle:
                    frame = pd.read_csv(handle, low_memory=False, na_values=["NAN", "nan", "NaN"])

                frame = frame.rename(columns=self.COLUMN_MAP)
                frame["cycle_start_time"] = pd.to_datetime(
                    frame["cycle_start_time"],
                    format="%Y-%m-%d %H:%M:%S",
                    errors="coerce",
                )
                for column in self.NUMERIC_COLUMNS:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")
                frame["mode_code"] = frame["mode_code"].fillna(0)

                asset_id = Path(member).stem
                asset_group = Path(member).parent.name
                frame = frame.sort_values("experiment_time_s").reset_index(drop=True)
                frame["cycle_id"] = pd.factorize(frame["cycle_start_time"])[0] + 1
                frame["sample_time_s"] = frame.groupby("cycle_id")["experiment_time_s"].transform(
                    lambda series: series - series.min()
                )
                frame["sample_timestamp"] = frame["cycle_start_time"] + pd.to_timedelta(
                    frame["sample_time_s"], unit="s"
                )
                frame["mode_label"] = frame["mode_code"].map(
                    {-1.0: "discharge", 0.0: "rest", 1.0: "charge"}
                )
                frame["mission_label"] = frame["mission_type_code"].map(
                    {0.0: "reference mission", 1.0: "regular mission"}
                )
                frame["voltage_v"] = frame["voltage_load_v"].combine_first(
                    frame["voltage_charger_v"]
                )
                frame["asset_id"] = asset_id
                frame["asset_group"] = asset_group
                frame["dataset_code"] = self.dataset_code
                frame["dataset_name"] = self.dataset_name
                frame["cycle_kind"] = self.cycle_kind
                frame["source_member"] = member
                frame["step_id"] = frame["cycle_id"]
                frame["step_comment"] = frame["mission_label"].fillna("unknown mission")
                frame["step_type"] = frame["mode_label"].fillna("unknown")
                frame["sample_index"] = np.arange(1, len(frame) + 1)
                frames.append(frame[_sample_columns()])

        return LoadedBatteryDataset(
            dataset_code=self.dataset_code,
            dataset_name=self.dataset_name,
            cycle_kind=self.cycle_kind,
            samples=(
                pd.concat(frames, ignore_index=True)
                if frames
                else pd.DataFrame(columns=_sample_columns())
            ),
        )


class RandomWalkLoader(BatteryDatasetLoader):
    dataset_code = "nasa_rw1"
    dataset_name = "Randomized Battery Usage 1: Random Walk"
    cycle_kind = "step"
    accepted_filenames = ("nasa_rw1_battery.zip",)

    def supports(self, zip_path: Path, members: list[str]) -> bool:
        has_expected_name = zip_path.name in self.accepted_filenames
        has_mat_members = any(member.endswith(".mat") for member in members)
        has_root = any(
            member.startswith("Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/")
            for member in members
        )
        return has_expected_name or (has_mat_members and has_root)

    def load(self, zip_path: Path, max_assets: int | None = None) -> LoadedBatteryDataset:
        frames: list[pd.DataFrame] = []
        with zipfile.ZipFile(zip_path) as archive:
            mat_members = [
                member
                for member in sorted(archive.namelist())
                if member.endswith(".mat") and "/data/Matlab/" in member
            ]
            if max_assets is not None:
                mat_members = mat_members[:max_assets]

            for member in mat_members:
                mat = loadmat(
                    io.BytesIO(archive.read(member)),
                    squeeze_me=True,
                    struct_as_record=False,
                )
                data = mat["data"]
                asset_id = Path(member).stem

                for step_index, step in enumerate(_flatten_steps(data.step), start=1):
                    relative_time = _coerce_float_vector(
                        getattr(step, "relativeTime", np.array([]))
                    )
                    experiment_time = _coerce_float_vector(getattr(step, "time", np.array([])))
                    voltage = _coerce_float_vector(getattr(step, "voltage", np.array([])))
                    current = _coerce_float_vector(getattr(step, "current", np.array([])))
                    temperature = _coerce_float_vector(getattr(step, "temperature", np.array([])))

                    vector_length = min(
                        len(relative_time),
                        len(experiment_time),
                        len(voltage),
                        len(current),
                        len(temperature),
                    )
                    if vector_length == 0:
                        continue

                    parsed_step_start = pd.to_datetime(
                        str(getattr(step, "date", "")),
                        format="%d-%b-%Y %H:%M:%S",
                        errors="coerce",
                    )
                    step_start_time = (
                        None if pd.isna(parsed_step_start) else pd.Timestamp(parsed_step_start)
                    )
                    type_code = str(getattr(step, "type", "R")).strip()[:1]
                    step_comment = str(getattr(step, "comment", "")).strip() or "unknown step"
                    frame = pd.DataFrame(
                        {
                            "dataset_code": self.dataset_code,
                            "dataset_name": self.dataset_name,
                            "asset_id": asset_id,
                            "asset_group": "random_walk",
                            "source_member": member,
                            "cycle_kind": self.cycle_kind,
                            "cycle_id": step_index,
                            "cycle_start_time": step_start_time,
                            "step_id": step_index,
                            "step_comment": step_comment,
                            "step_type": _rw_type_label(type_code),
                            "mode_code": _rw_mode_code(type_code),
                            "sample_index": np.arange(1, vector_length + 1),
                            "sample_time_s": relative_time[:vector_length],
                            "experiment_time_s": experiment_time[:vector_length],
                            "sample_timestamp": _timestamp_series(
                                step_start_time,
                                relative_time[:vector_length],
                            ),
                            "voltage_v": voltage[:vector_length],
                            "current_a": current[:vector_length],
                            "temperature_c": temperature[:vector_length],
                            "voltage_charger_v": np.nan,
                            "voltage_load_v": np.nan,
                            "temperature_mosfet_c": np.nan,
                            "temperature_resistor_c": np.nan,
                            "mission_type_code": pd.NA,
                            "mission_label": step_comment,
                        }
                    )
                    frames.append(frame[_sample_columns()])

        return LoadedBatteryDataset(
            dataset_code=self.dataset_code,
            dataset_name=self.dataset_name,
            cycle_kind=self.cycle_kind,
            samples=(
                pd.concat(frames, ignore_index=True)
                if frames
                else pd.DataFrame(columns=_sample_columns())
            ),
        )


LOADERS: tuple[BatteryDatasetLoader, ...] = (
    RandomizedRecommissionedLoader(),
    RandomWalkLoader(),
)


def preprocess_dataset(
    input_path: Path | str | None = None,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    max_assets: int | None = None,
) -> dict[str, Path]:
    """Load a supported NASA battery archive and emit normalized parquet outputs."""
    resolved_input = Path(input_path) if input_path is not None else detect_default_input()
    resolved_output = Path(output_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)

    loader = resolve_loader(resolved_input)
    dataset = loader.load(resolved_input, max_assets=max_assets)
    samples = dataset.samples.sort_values(
        ["dataset_code", "asset_id", "cycle_id", "sample_time_s", "sample_index"]
    ).reset_index(drop=True)
    cycle_features = build_cycle_features(samples)
    incident_windows = build_incident_windows(samples)
    manifest = pd.DataFrame(
        [
            {
                "dataset_code": dataset.dataset_code,
                "dataset_name": dataset.dataset_name,
                "input_zip": str(resolved_input),
                "asset_count": int(samples["asset_id"].nunique()) if not samples.empty else 0,
                "sample_count": int(len(samples)),
                "cycle_count": int(len(cycle_features)),
                "incident_window_count": int(len(incident_windows)),
            }
        ]
    )

    outputs = {
        "telemetry_samples": resolved_output / "telemetry_samples.parquet",
        "cycle_features": resolved_output / "cycle_features.parquet",
        "incident_windows": resolved_output / "incident_windows.parquet",
        "preprocess_manifest": resolved_output / "preprocess_manifest.parquet",
    }
    samples.to_parquet(outputs["telemetry_samples"], index=False)
    cycle_features.to_parquet(outputs["cycle_features"], index=False)
    incident_windows.to_parquet(outputs["incident_windows"], index=False)
    manifest.to_parquet(outputs["preprocess_manifest"], index=False)
    return outputs


def detect_default_input(raw_dir: Path = DEFAULT_RAW_DIR) -> Path:
    """Pick the preferred local archive if one of the accepted inputs exists."""
    for filename in EXPECTED_INPUTS:
        candidate = raw_dir / filename
        if candidate.exists():
            return candidate

    expected_paths = ", ".join(str(raw_dir / filename) for filename in EXPECTED_INPUTS)
    raise FileNotFoundError(
        "No supported NASA battery archive was found. "
        f"Expected one of: {expected_paths}"
    )


def resolve_loader(zip_path: Path) -> BatteryDatasetLoader:
    """Inspect the archive structure and choose a compatible loader."""
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()

    for loader in LOADERS:
        if loader.supports(zip_path, members):
            return loader

    raise ValueError(f"Unsupported battery archive structure: {zip_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess a supported NASA battery ZIP archive.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to nasa_rr_battery.zip or nasa_rw1_battery.zip. Defaults to data/raw/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where parquet outputs will be written.",
    )
    parser.add_argument(
        "--max-assets",
        type=int,
        default=None,
        help="Optional cap on assets processed, useful for smoke runs on the full archive.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = preprocess_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        max_assets=args.max_assets,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")


def _flatten_steps(step_container: Any) -> list[Any]:
    if isinstance(step_container, np.ndarray):
        return [step for step in step_container.flat]
    return [step_container]


def _coerce_float_vector(values: Any) -> np.ndarray:
    return np.atleast_1d(np.asarray(values, dtype=float))


def _rw_mode_code(type_code: str) -> int:
    return {"D": -1, "R": 0, "C": 1}.get(type_code.upper(), 0)


def _rw_type_label(type_code: str) -> str:
    return {"D": "discharge", "R": "rest", "C": "charge"}.get(type_code.upper(), "unknown")


def _timestamp_series(
    start_time: pd.Timestamp | None,
    relative_time: np.ndarray,
) -> pd.Series:
    if start_time is None:
        return pd.Series(pd.NaT, index=np.arange(len(relative_time)))
    return pd.Series(start_time + pd.to_timedelta(relative_time, unit="s"))


def _sample_columns() -> list[str]:
    return [
        "dataset_code",
        "dataset_name",
        "asset_id",
        "asset_group",
        "source_member",
        "cycle_kind",
        "cycle_id",
        "cycle_start_time",
        "step_id",
        "step_comment",
        "step_type",
        "mode_code",
        "sample_index",
        "sample_time_s",
        "experiment_time_s",
        "sample_timestamp",
        "voltage_v",
        "current_a",
        "temperature_c",
        "voltage_charger_v",
        "voltage_load_v",
        "temperature_mosfet_c",
        "temperature_resistor_c",
        "mission_type_code",
        "mission_label",
    ]


if __name__ == "__main__":
    main()
