import numpy as np
import pandas as pd
from dlr_planb.physics import ampacity
from dlr_planb import config

def metrics(a, p):
    """Basic error metrics."""
    m = np.isfinite(a) & np.isfinite(p) & (a > 0)
    a, p = a[m], p[m]
    if len(a) == 0:
        return np.nan, np.nan, np.nan, np.nan
    e = p - a
    rng = np.max(a) - np.min(a) if len(a) > 1 else 1.0
    return np.mean(np.abs(e)), np.sqrt(np.mean(e ** 2)), np.mean(np.abs(e) / a) * 100, np.sqrt(np.mean(e ** 2)) / rng

def static_rating_conventional():
    """B-8: Conventional static rating (0.6 m/s wind perpendicular, 35C ambient, 1000 W/m2 sun)"""
    return ampacity(Ta=35.0, V=0.6, wdir=90.0, ghi=1000.0, azimuth=config.LINE_AZIMUTH_DEG, Tc=config.T_MAX)

def safety_metrics(a, p10, p50, p90, static_proxy):
    """B-7: Safety metrics (P10 violation, pinball, etc.)"""
    m = np.isfinite(a) & np.isfinite(p10) & np.isfinite(p90)
    a, p10, p50, p90 = a[m], p10[m], p50[m], p90[m]
    
    # Non-crossing check
    crossing_issues = np.mean((p10 > p50) | (p50 > p90)) * 100
    
    # Coverage & Violations
    coverage = np.mean((a >= p10) & (a <= p90)) * 100
    p10_too_high = np.mean(p10 > a) * 100 # Unsafe condition
    p10_violation_severity = np.mean(np.maximum(p10 - a, 0)) # How bad are the violations?
    
    # Pinball loss
    pinball = np.mean([
        np.mean(np.maximum(q * (a - p), (q - 1) * (a - p)))
        for q, p in [(0.1, p10), (0.5, p50), (0.9, p90)]
    ])
    
    return {
        "Coverage P10-P90 (%)": coverage,
        "P10 Unsafe (%)": p10_too_high,
        "P10 Violation Severity (A)": p10_violation_severity,
        "Avg Pinball Loss": pinball,
        "Crossing Issues (%)": crossing_issues,
        "P10 avg gain over static (A)": np.mean(p10) - static_proxy
    }

def performance_breakdowns(df_results):
    """B-9: Breakdowns by month, hour of day, and wind speed bin."""
    # df_results expects columns: Time, Actual, Pred_Main, wind
    df = df_results.copy()
    df['Month'] = df['Time'].dt.month
    df['Hour'] = df['Time'].dt.hour
    df['WindBin'] = pd.cut(df['wind'], bins=[0, 2, 5, 10, 30], labels=['0-2', '2-5', '5-10', '10+'])
    
    breakdowns = {}
    for col in ['Month', 'Hour', 'WindBin']:
        res = df.groupby(col).apply(lambda g: pd.Series({
            'MAE': np.mean(np.abs(g['Pred_Main'] - g['Actual'])),
            'Count': len(g)
        }))
        breakdowns[col] = res
    return breakdowns
