from __future__ import annotations

import pandas as pd


class FeatureBuilder:
    def build_frame(self, telemetry: pd.DataFrame, targets: pd.DataFrame | None = None) -> pd.DataFrame:
        if "date" not in telemetry.columns:
            raise ValueError("telemetry must include 'date'")
        frame = telemetry.sort_values("date").copy()
        if targets is not None and not targets.empty:
            from neftecode.data.sync import merge_asof_on_date

            frame = merge_asof_on_date(frame, targets)
        return frame.reset_index(drop=True)
