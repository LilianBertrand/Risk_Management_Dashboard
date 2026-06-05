"""Risk report generation and export helpers."""

from __future__ import annotations

from io import BytesIO

import pandas as pd


def create_summary_report(metrics: dict) -> pd.DataFrame:
    """Create a one-column risk report from a metrics dictionary."""
    return pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())})


def create_excel_report(sheets: dict[str, pd.DataFrame]) -> bytes:
    """Create an in-memory Excel workbook from multiple DataFrames."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            safe_name = sheet_name[:31]
            df.to_excel(writer, index=True, sheet_name=safe_name)
    return output.getvalue()
