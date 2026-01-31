import numpy as np

import pandas as pd

def speedGPS2Water(v, heading_ship, cu, cv):
    """将GPS速度转换为水动力速度"""
    return v - cu * np.sin(np.deg2rad(heading_ship)) - cv * np.cos(np.deg2rad(heading_ship))

def Rtotal2Pe(row):
    """计算推进功率（单位：千牛）"""
    cu = row['stream_val'] * np.sin(np.deg2rad(row['stream_direction']))
    cv = row['stream_val'] * np.cos(np.deg2rad(row['stream_direction']))
    sog = row['SOG'] * 0.5144  # 节转换为 m/s
    return speedGPS2Water(sog, row['heading'], cu, cv) * row['R_t_pre']

def calFuelHour(row):
    """计算每小时燃油消耗（单位：吨/小时）"""
    eta = 1.0 * 0.60 * 0.99 * 1.1
    return (row['P_pre'] / eta) * 200 / 1e6

def compute_daily_fuel(
    df,
    time_col="postime",
    emission_factor=None,
):
    """
    Compute hourly fuel consumption and aggregate it to daily fuel.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame that MUST contain:
        - R_a_pre   : predicted added resistance
        - R_calm    : calm water resistance
        - SOG, heading, stream_val, stream_direction
        - num       : vessel identifier
        - time_col  : timestamp column (default: 'postime')
    time_col : str, optional
        Name of the timestamp column, by default 'postime'.
    emission_factor : float or None, optional
        If provided, daily CO2 emissions will be computed as:
        Predict_CO2 = Predict_fuel * emission_factor.

    Returns
    -------
    df_hourly : pandas.DataFrame
        Row-level DataFrame with:
        - R_t_pre
        - P_pre
        - Fuel_hour
    df_daily : pandas.DataFrame
        Daily aggregated fuel consumption (and CO2 if specified).
    """

    df_hourly = df.copy()

    # --------------------------------------------------
    # 1. Total resistance
    # --------------------------------------------------
    df_hourly["R_t_pre"] = df_hourly["R_a_pre"] + df_hourly["R_calm"]

    # --------------------------------------------------
    # 2. Propulsion power
    # --------------------------------------------------
    df_hourly["P_pre"] = df_hourly.apply(Rtotal2Pe, axis=1)

    # --------------------------------------------------
    # 3. Hourly fuel consumption
    # --------------------------------------------------
    df_hourly["Fuel_hour"] = df_hourly.apply(calFuelHour, axis=1)

    # --------------------------------------------------
    # 4. Daily aggregation
    # --------------------------------------------------
    df_hourly["date"] = pd.to_datetime(df_hourly[time_col]).dt.date

    df_daily = (
        df_hourly
        .groupby(["Vessel ID", "date"])
        .agg(Predict_fuel=("Fuel_hour", "sum"))
        .reset_index()
    )

    # --------------------------------------------------
    # 5. Optional CO2 emissions
    # --------------------------------------------------
    if emission_factor is not None:
        df_daily["emission_predict_day"] = df_daily["Predict_fuel"] * emission_factor

    return df_hourly, df_daily