import numpy as np
import pandas as pd
from dlr_planb.physics import ampacity
from dlr_planb import config
from dlr_planb.data import C4

def build_features(Wd, k, with_fc=True, is_train=False):
    L = 24 * k; y = Wd["Ampacity"]; f = {}
    for lag in [0, 1, 2, 3, 6, 12, 24, 48, 168]:
        f[f"y_lag{lag}"] = y.shift(L + lag)
    f["y_mean24"] = y.rolling(24).mean().shift(L); f["y_mean168"] = y.rolling(168).mean().shift(L); f["y_std24"] = y.rolling(24).std().shift(L)
    for c in C4:
        f[f"{c}_issue"] = Wd[c].shift(L); f[f"{c}_mean24"] = Wd[c].rolling(24).mean().shift(L)
        if c == 'dir':
            f['dir_sin_issue'] = np.sin(np.radians(Wd[c].shift(L)))
            f['dir_cos_issue'] = np.cos(np.radians(Wd[c].shift(L)))
            
    hr, doy = Wd["lh"].to_numpy(), Wd.index.dayofyear.to_numpy()
    f["hour_sin"] = np.sin(2 * np.pi * hr / 24); f["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365); f["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    
    # B-4: Leave-one-out climatology for train rows
    if is_train:
        # compute leave-one-out clim
        sum_by_group = Wd.groupby(["lh", "month"])["Ampacity"].transform('sum')
        count_by_group = Wd.groupby(["lh", "month"])["Ampacity"].transform('count')
        # to prevent division by zero or negative counts, fallback to overall mean
        f["clim"] = (sum_by_group - Wd["Ampacity"]) / (count_by_group - 1).clip(lower=1)
    else:
        f["clim"] = Wd["clim"].to_numpy()
        
    if with_fc:
        for c in C4:
            f[f"fc_{c}"] = Wd[f"{c}_L{k}"]
            if c == 'dir':
                f[f"fcerr_{c}_issue"] = (Wd[f"{c}_L{k}"] - Wd[c] + 180).shift(L).mod(360) - 180
                f['fc_dir_sin'] = np.sin(np.radians(Wd[f"{c}_L{k}"]))
                f['fc_dir_cos'] = np.cos(np.radians(Wd[f"{c}_L{k}"]))
            else:
                f[f"fcerr_{c}_issue"] = (Wd[f"{c}_L{k}"] - Wd[c]).shift(L)     # error of the forecast for the issue hour: already verified at issue time
        f["fc_phys"] = ampacity(Wd[f"temp_L{k}"], Wd[f"wind_L{k}"], Wd[f"dir_L{k}"], Wd[f"ghi_L{k}"], azimuth=config.LINE_AZIMUTH_DEG, Tc=config.T_MAX)
        if "clim" in Wd.columns:
            f["fc_phys_minus_clim"] = f["fc_phys"] - Wd["clim"].to_numpy()
    return pd.DataFrame(f, index=Wd.index)
