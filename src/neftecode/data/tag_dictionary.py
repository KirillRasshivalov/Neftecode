from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class TagInfo:
    tag: str
    description: str | None = None
    unit: str | None = None
    sheet: str | None = None


class TagDictionary:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._by_tag: dict[str, TagInfo] = {}

    def load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        xls = pd.ExcelFile(self.path)
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)
            tag_col = self._guess_tag_column(df)
            if tag_col is None:
                continue
            for _, row in df.iterrows():
                tag = str(row[tag_col]).strip()
                if not tag or tag.lower() == "nan":
                    continue
                self._by_tag[tag] = TagInfo(
                    tag=tag,
                    description=str(row.get("description", row.get("описание", ""))) or None,
                    unit=str(row.get("unit", row.get("ед", ""))) or None,
                    sheet=sheet,
                )

    def get(self, tag: str) -> TagInfo | None:
        return self._by_tag.get(tag)

    def require(self, tag: str) -> TagInfo:
        info = self.get(tag)
        if info is None:
            raise KeyError(f"Tag '{tag}' not found in dictionary — do not infer meaning from short name")
        return info

    @staticmethod
    def _guess_tag_column(df: pd.DataFrame) -> str | None:
        for candidate in ("tag", "Tag", "ТЕГ", "тег", "name", "Name"):
            if candidate in df.columns:
                return candidate
        return df.columns[0] if len(df.columns) else None
