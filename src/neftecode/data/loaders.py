from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from neftecode.data.config import project_root


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @classmethod
    def default(cls) -> DataPaths:
        return cls(root=project_root() / "data")

    @property
    def avt_tags(self) -> Path:
        return self.root / "avt_tags.csv"

    @property
    def hydro_tags(self) -> Path:
        return self.root / "242000_tags.csv"

    @property
    def lims(self) -> Path:
        return self.root / "lims.xlsx"

    @property
    def pak(self) -> Path:
        return self.root / "pak.xlsx"

    @property
    def tag_dictionary(self) -> Path:
        return self.root / "tags_dictionary.xlsx"


class TelemetryStore:
    def __init__(self, paths: DataPaths | None = None) -> None:
        self.paths = paths or DataPaths.default()
        self._avt: pd.DataFrame | None = None
        self._hydro: pd.DataFrame | None = None

    def load_avt(self) -> pd.DataFrame:
        if self._avt is None:
            self._avt = self._read_telemetry_csv(self.paths.avt_tags)
        return self._avt

    def load_hydro(self) -> pd.DataFrame:
        if self._hydro is None:
            self._hydro = self._read_telemetry_csv(self.paths.hydro_tags)
        return self._hydro

    @staticmethod
    def _read_telemetry_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame(columns=["date"])
        df = pd.read_csv(path)
        if "date" not in df.columns:
            raise ValueError(f"{path} must contain a 'date' column")
        df["date"] = pd.to_datetime(df["date"])
        unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
        if unnamed:
            df = df.drop(columns=unnamed)
        return df.sort_values("date").reset_index(drop=True)
