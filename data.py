import os
import time
import requests
import numpy as np
import pandas as pd
from dlr_planb import config
from dlr_planb.physics import ampacity

C4 = ["temp", "wind", "dir", "ghi"]
BASE = {"temp": "temperature_2m", "wind": "wind_speed_10m", "dir": "wind_direction_10m", "ghi": "shortwave_radiation"}
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PREV_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

def fetch_json(url, params, tries=5):
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=120)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(30 * (k + 1)); continue
            raise RuntimeError(f"HTTP {r.status_code} from {r.url}\n{r.text[:600]}")
        except requests.exceptions.RequestException as e:
            if k == tries - 1:
                raise RuntimeError(f"Failed to fetch {url}: {e}")
            time.sleep(10 * (k + 1))
    raise RuntimeError("Open-Meteo rate limit or connection issues: wait a few minutes and rerun.")

def date_chunks(start, end, days=120):
    s, e, out = pd.Timestamp(start), pd.Timestamp(end), []
    while s <= e:
        c = min(s + pd.Timedelta(days=days - 1), e); out.append((s.date().isoformat(), c.date().isoformat())); s = c + pd.Timedelta(days=1)
    return out

def hourly_frame(js, wanted):                 # wanted: {new_name: base_variable_name}
    h = js["hourly"]; idx = pd.to_datetime(h["time"]); out = {}
    for new, base in wanted.items():
        key = next((k for k in h if k == base or k.startswith(base)), None)
        out[new] = np.array(h[key], dtype=float) if key else np.full(len(idx), np.nan)
    return pd.DataFrame(out, index=idx)

def build_w():
    if os.path.exists("om_actual.csv") and os.path.exists("om_fc.csv") and not config.REFRESH:
        act = pd.read_csv("om_actual.csv", index_col=0, parse_dates=True); fc = pd.read_csv("om_fc.csv", index_col=0, parse_dates=True)
        print("Loaded cached downloads.")
    else:
        act_parts, fc_parts = [], []
        for s, e in date_chunks(config.START_DATE, config.END_DATE):
            p = dict(latitude=config.LAT, longitude=config.LON, start_date=s, end_date=e, hourly=",".join(BASE.values()), wind_speed_unit="ms", timezone="GMT")
            act_parts.append(hourly_frame(fetch_json(ARCHIVE_URL, p), {c: BASE[c] for c in C4}))
            wanted = {f"{c}_L{k}": f"{BASE[c]}_previous_day{k}" for c in C4 for k in config.LEADS}
            p2 = dict(latitude=config.LAT, longitude=config.LON, start_date=s, end_date=e, wind_speed_unit="ms", timezone="GMT",
                      hourly=",".join(wanted.values()))
            if config.MODEL: p2["models"] = config.MODEL
            fc_parts.append(hourly_frame(fetch_json(PREV_URL, p2), wanted))
            print(f"  downloaded {s} -> {e}"); time.sleep(1.0)
        act = pd.concat(act_parts); fc = pd.concat(fc_parts)
        act = act[~act.index.duplicated()]; fc = fc[~fc.index.duplicated()]
        act.to_csv("om_actual.csv"); fc.to_csv("om_fc.csv")

    W = act.join(fc, how="inner").dropna()
    
    # Gap check (B-3)
    full_idx = pd.date_range(start=W.index.min(), end=W.index.max(), freq='h')
    try:
        assert len(full_idx) == len(W), "Index is not a complete hourly range"
    except AssertionError:
        W = W.reindex(full_idx)

    W["Ampacity"] = ampacity(W["temp"], W["wind"], W["dir"], W["ghi"], azimuth=config.LINE_AZIMUTH_DEG, Tc=config.T_MAX)
    W["lh"] = (W.index + pd.Timedelta(hours=config.LOCAL_UTC_OFFSET)).hour
    W["month"] = W.index.month
    return W
