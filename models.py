import numpy as np
import pandas as pd
import lightgbm as lgb
from dlr_planb import config

def train_gbm(X_train, y_train, X_val, y_val, clim_train, clim_val, is_quantile=False, alpha=None):
    """
    Train LightGBM model. 
    If is_quantile is False, trains the main regression model (L1 objective).
    If is_quantile is True, trains a quantile regression model for the given alpha.
    """
    if is_quantile:
        if alpha is None:
            raise ValueError("alpha must be provided for quantile regression")
        # Quantile model from cell 7
        # objective="quantile", n_estimators=400, learning_rate=0.04
        # Same other params
        mdl = lgb.LGBMRegressor(
            objective="quantile", alpha=alpha, n_estimators=400, learning_rate=0.04,
            num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.6, reg_lambda=5.0, random_state=config.SEED, verbose=-1
        )
        # Note: cell 7 fits without early stopping and on direct target y
        mdl.fit(X_train, y_train)
        return mdl
    else:
        # Standard regression model from cell 6
        # objective="regression_l1", n_estimators=3000, learning_rate=0.03, num_leaves=15, min_child_samples=40
        # subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0, random_state=0
        # early_stopping(150)
        mdl = lgb.LGBMRegressor(
            objective="regression_l1", n_estimators=3000, learning_rate=0.03,
            num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.6, reg_lambda=5.0, random_state=config.SEED, verbose=-1
        )
        mdl.fit(
            X_train, y_train - clim_train, 
            eval_set=[(X_val, y_val - clim_val)], 
            eval_metric="l1",
            callbacks=[lgb.early_stopping(150, verbose=False)]
        )
        return mdl

def predict_gbm(mdl, X, clim=0.0, is_quantile=False):
    if is_quantile:
        return mdl.predict(X)
    else:
        return mdl.predict(X) + clim

def calibrate_weather(W_train, W_test, lead):
    """
    B-1: Calibrate weather forecasts, notably circular wind direction using sin/cos.
    Returns a dataframe of calibrated weather for W_test.
    """
    calibrated = pd.DataFrame(index=W_test.index)
    X_train = W_train[[f"temp_L{lead}", f"wind_L{lead}", f"ghi_L{lead}"]].copy()
    X_train['dir_sin'] = np.sin(np.radians(W_train[f"dir_L{lead}"]))
    X_train['dir_cos'] = np.cos(np.radians(W_train[f"dir_L{lead}"]))
    
    X_test = W_test[[f"temp_L{lead}", f"wind_L{lead}", f"ghi_L{lead}"]].copy()
    X_test['dir_sin'] = np.sin(np.radians(W_test[f"dir_L{lead}"]))
    X_test['dir_cos'] = np.cos(np.radians(W_test[f"dir_L{lead}"]))
    
    targets = {
        'temp': W_train['temp'],
        'wind': W_train['wind'],
        'ghi': W_train['ghi'],
        'dir_sin_act': np.sin(np.radians(W_train['dir'])),
        'dir_cos_act': np.cos(np.radians(W_train['dir']))
    }
    
    preds = {}
    for t_name, y_tr in targets.items():
        mdl = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=config.SEED, verbose=-1)
        mdl.fit(X_train, y_tr)
        preds[t_name] = mdl.predict(X_test)
        
    calibrated['temp'] = preds['temp']
    calibrated['wind'] = np.clip(preds['wind'], 0, None)
    calibrated['ghi'] = np.clip(preds['ghi'], 0, None)
    
    # B-1: Recombine dir using arctan2
    calibrated['dir'] = np.degrees(np.arctan2(preds['dir_sin_act'], preds['dir_cos_act'])) % 360
    
    return calibrated
