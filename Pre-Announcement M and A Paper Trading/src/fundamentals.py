from __future__ import annotations

import numpy as np
import pandas as pd

from .point_in_time import merge_asof_point_in_time


def add_cash_runway(fundamentals: pd.DataFrame) -> pd.DataFrame:
    df = fundamentals.copy()
    cash = df.get("cash", 0).fillna(0) + df.get("cash_equivalents", 0).fillna(0)
    ocf = df.get("operating_cash_flow", np.nan)
    burn = (-ocf).clip(lower=0)
    df["cash_and_equivalents"] = cash
    df["cash_burn_quarter"] = burn
    df["cash_runway_months"] = np.where(burn > 0, cash / burn * 3.0, np.inf)
    df["working_capital"] = df.get("current_assets", np.nan) - df.get("current_liabilities", np.nan)
    df["current_ratio"] = df.get("current_assets", np.nan) / df.get("current_liabilities", np.nan).replace(0, np.nan)
    df["cash_debt_ratio"] = cash / df.get("total_debt", np.nan).replace(0, np.nan)
    return df


def point_in_time_fundamentals(observations: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    f = add_cash_runway(fundamentals)
    return merge_asof_point_in_time(
        observations,
        f,
        by=["cik"],
        obs_ts_col="prediction_ts",
        source_ts_col="information_available_timestamp",
    )
