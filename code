#!/usr/bin/env python3
"""
INDENG 290 Energy Analytics — HW01 Forecasting
Reproducible script that replicates the logic used in the submitted Colab notebook.

Expected files in the same directory as this script:
- CAISOHourlyLoadCSV.csv

Outputs:
- Q1(a) March 2026 peak load point forecast (MW)
- Q1(b) Forecasted date of March 2026 peak load
- Q1(c) Monthly peak backtest + MAE/RMSE/MAPE
- Q3(c) (extra credit) 24 shape factors for 2025-03-14
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_DATA_PATH = "CAISOHourlyLoadCSV.csv"
DEFAULT_GROWTH_RATE = 0.015
DEFAULT_PEAK_DAY = "2025-03-14"


@dataclass
class LoadData:
    df: pd.DataFrame
    load_col: str


def infer_load_col(df: pd.DataFrame) -> str:
    """Infer the load column name robustly."""
    candidates = ["CAISO Load (MW)", "Load_MW", "Load MW", "Load", "MW", "load_mw", "CAISO_Load_MW"]
    for c in candidates:
        if c in df.columns:
            return c

    # Fallback: pick first numeric column not obviously metadata
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c.lower() not in {"hour", "year", "month", "day"}]
    if not numeric_cols:
        raise ValueError(
            "Could not infer load column. Please ensure your CSV has a load column like "
            "'CAISO Load (MW)' or rename accordingly."
        )
    return numeric_cols[0]


def load_dataset(path: str) -> LoadData:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if "Date" not in df.columns or "Hour" not in df.columns:
        raise ValueError("CSV must contain columns 'Date' and 'Hour'.")

    df["Date"] = pd.to_datetime(df["Date"])
    df["Hour"] = df["Hour"].astype(int)

    # Add common fields used in notebook
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    load_col = infer_load_col(df)
    return LoadData(df=df, load_col=load_col)


def march_peak(df: pd.DataFrame, load_col: str, year: int) -> Tuple[float, pd.Timestamp, int]:
    """Return (peak_mw, peak_date_timestamp, peak_hour) for March of given year."""
    march = df[(df["Year"] == year) & (df["Month"] == 3)].copy()
    if march.empty:
        raise ValueError(f"No March data found for year {year}.")
    peak_idx = march[load_col].idxmax()
    peak_row = march.loc[peak_idx]
    return float(peak_row[load_col]), pd.Timestamp(peak_row["Date"]), int(peak_row["Hour"])


def backtest_monthly_peaks(df: pd.DataFrame, load_col: str, min_train_months: int = 3) -> pd.DataFrame:
    """Rolling-origin backtest on monthly peak MW using linear trend on prior monthly peaks."""
    tmp = df.copy()
    tmp["YearMonth"] = tmp["Date"].dt.to_period("M").astype(str)

    monthly_peaks = (
        tmp.groupby("YearMonth", as_index=False)[load_col]
        .max()
        .rename(columns={load_col: "Peak_MW"})
    )

    monthly_peaks["t"] = np.arange(len(monthly_peaks))
    rows = []

    for i in range(min_train_months, len(monthly_peaks)):
        train = monthly_peaks.iloc[:i].copy()
        test = monthly_peaks.iloc[i].copy()

        X = train["t"].values
        y = train["Peak_MW"].values

        b, a = np.polyfit(X, y, 1)  # y = a + b*t
        yhat = a + b * test["t"]

        rows.append(
            {
                "YearMonth": test["YearMonth"],
                "Actual_Peak_MW": float(test["Peak_MW"]),
                "Forecast_Peak_MW": float(yhat),
                "Error_MW": float(yhat - test["Peak_MW"]),
                "AbsError_MW": float(abs(yhat - test["Peak_MW"])),
                "APE_pct": float(abs(yhat - test["Peak_MW"]) / test["Peak_MW"] * 100.0),
            }
        )

    return pd.DataFrame(rows)


def shape_factors_for_day(df: pd.DataFrame, load_col: str, day: str) -> pd.DataFrame:
    """Compute 24 shape factors for a given date (YYYY-MM-DD)."""
    peak_day = pd.Timestamp(day)
    day_df = df[df["Date"] == peak_day].copy()
    if day_df.empty:
        raise ValueError(f"No rows found for date {day}. Check date format and dataset coverage.")

    L_star = day_df[load_col].max()
    day_df["shape_factor"] = day_df[load_col] / L_star

    out = day_df.sort_values("Hour")[["Hour", "shape_factor"]].reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to CAISO hourly load CSV.")
    parser.add_argument("--growth", type=float, default=DEFAULT_GROWTH_RATE, help="Growth rate for March 2026 peak forecast.")
    parser.add_argument("--peakday", default=DEFAULT_PEAK_DAY, help="Date to compute shape factors for (YYYY-MM-DD).")
    parser.add_argument("--save-backtest", default="", help="Optional path to save backtest results as CSV.")
    args = parser.parse_args()

    data = load_dataset(args.data)
    df, load_col = data.df, data.load_col

    # ---- Q1(a) + Q1(b): Peak forecast and date ----
    L_2025, d_2025_ts, h_2025 = march_peak(df, load_col, year=2025)
    L_2026 = L_2025 * (1.0 + args.growth)
    d_star = d_2025_ts.date().replace(year=2026)

    print("=== Q1(a) March 2026 CAISO Peak Load Point Forecast ===")
    print(f"March 2025 peak load: {L_2025:,.2f} MW")
    print(f"Peak occurred on {d_2025_ts.date()}, Hour {h_2025} (hour-ending)")
    print(f"March 2026 peak load forecast (unrounded): {L_2026:,.2f} MW")
    print(f"March 2026 peak load forecast (rounded): {int(round(L_2026)):,} MW")
    print()

    print("=== Q1(b) Forecasted Date of March 2026 CAISO Peak Load ===")
    print(f"March 2025 peak date: {d_2025_ts.date()}")
    print(f"Forecasted March 2026 peak date: {d_star}")
    print()

    # ---- Q1(c): Backtest ----
    bt = backtest_monthly_peaks(df, load_col, min_train_months=3)
    if bt.empty:
        print("Backtest could not be computed (not enough months).")
    else:
        mae = bt["AbsError_MW"].mean()
        rmse = float(np.sqrt(np.mean((bt["Error_MW"].values) ** 2)))
        mape = bt["APE_pct"].mean()

        print("=== Q1(c) Monthly Peak Backtest (rolling-origin, linear trend) ===")
        print(f"Backtest months: {bt.shape[0]}")
        print(f"MAE (MW): {mae:,.1f}")
        print(f"RMSE (MW): {rmse:,.1f}")
        print(f"MAPE (%): {mape:,.2f}")
        print()

        # Print a small preview (avoid dumping giant tables)
        print("Backtest preview (first 10 rows):")
        print(bt.head(10).to_string(index=False))
        print()

        if args.save_backtest:
            bt.to_csv(args.save_backtest, index=False)
            print(f"Saved backtest results to: {args.save_backtest}")
            print()

    # ---- Q3(c) Extra credit: shape factors ----
    print("=== Q3(c) 24 Shape Factors (observed profile on peak day) ===")
    sf = shape_factors_for_day(df, load_col, args.peakday)
    print(f"Peak-day: {args.peakday}")
    print("Max shape factor:", f"{sf['shape_factor'].max():.2f}")
    print(sf.to_string(index=False))


if __name__ == "__main__":
    main()
