import os
import numpy as np
import pandas as pd
from dlr_planb import config
from dlr_planb.data import build_w
from dlr_planb.features import build_features
from dlr_planb.models import train_gbm, predict_gbm, calibrate_weather
from dlr_planb.evaluate import metrics, safety_metrics, performance_breakdowns, static_rating_conventional
from dlr_planb.physics import ampacity

def main():
    print("Loading data...")
    W = build_w()
    n = len(W)
    i1, i2 = int(n * 0.6), int(n * 0.8) # Train 60%, Val 20%, Test 20%
    y = W["Ampacity"].to_numpy()
    
    # Compute overall climatology from train set and add to W
    train_W = W.iloc[:i1]
    clim_map = train_W.groupby(["lh", "month"])["Ampacity"].mean()
    W["clim"] = W.set_index(["lh", "month"]).index.map(clim_map).fillna(train_W["Ampacity"].mean())
    
    os.makedirs("results", exist_ok=True)
    static_conventional = static_rating_conventional()
    print(f"Static rating (conventional): {static_conventional:.1f} A")
    
    for k in config.LEADS:
        print(f"\n{'='*40}")
        print(f"Processing lead time {k} (x24 hours)")
        
        # Build features
        Xf_train = build_features(W.iloc[:i1], k, with_fc=True, is_train=True)
        Xf_val = build_features(W.iloc[:i2], k, with_fc=True, is_train=False).iloc[i1:i2]
        Xf_test = build_features(W, k, with_fc=True, is_train=False).iloc[i2:]
        
        ok_tr, ok_va, ok_te = Xf_train.notna().all(axis=1), Xf_val.notna().all(axis=1), Xf_test.notna().all(axis=1)
        
        X_tr, y_tr, clim_tr = Xf_train[ok_tr], y[:i1][ok_tr], Xf_train["clim"][ok_tr].to_numpy()
        X_va, y_va, clim_va = Xf_val[ok_va], y[i1:i2][ok_va], Xf_val["clim"][ok_va].to_numpy()
        X_te, y_te, clim_te = Xf_test[ok_te], y[i2:][ok_te], Xf_test["clim"][ok_te].to_numpy()
        
        print("Training main regression model...")
        mdl = train_gbm(X_tr, y_tr, X_va, y_va, clim_tr, clim_va, is_quantile=False)
        preds = predict_gbm(mdl, X_te, clim=clim_te, is_quantile=False)
        
        print("Training quantile models (0.1, 0.5, 0.9)...")
        q_preds = {}
        for q in [0.1, 0.5, 0.9]:
            q_mdl = train_gbm(X_tr, y_tr, X_va, y_va, None, None, is_quantile=True, alpha=q)
            q_preds[q] = predict_gbm(q_mdl, X_te, is_quantile=True)
            
        # Conformal calibration on validation set (B-7)
        print("Applying conformal calibration on validation set...")
        q_val_preds = {}
        for q in [0.1, 0.9]:
            q_mdl = train_gbm(X_tr, y_tr, X_va, y_va, None, None, is_quantile=True, alpha=q)
            q_val_preds[q] = predict_gbm(q_mdl, X_va, is_quantile=True)
        
        c_low = np.quantile(q_val_preds[0.1] - y_va, 0.9)
        c_up = np.quantile(y_va - q_val_preds[0.9], 0.9)
        
        p10 = q_preds[0.1] - c_low
        p50 = q_preds[0.5]
        p90 = q_preds[0.9] + c_up
        
        res_df = pd.DataFrame({
            "Time": W.index[i2:][ok_te],
            "Actual": y_te,
            "Pred_Main": preds,
            "P10": p10,
            "P50": p50,
            "P90": p90,
            "wind": W.iloc[i2:][ok_te]["wind"].values
        })
        
        # B-1: Weather Calibration
        print("Calibrating weather features...")
        cal_weather = calibrate_weather(W.iloc[:i1][ok_tr], W.iloc[i2:][ok_te], k)
        res_df["Pred_Phys_Cal"] = ampacity(
            cal_weather['temp'], cal_weather['wind'], cal_weather['dir'], cal_weather['ghi'], 
            azimuth=config.LINE_AZIMUTH_DEG, Tc=config.T_MAX
        )
        
        res_df.to_csv(f"results/predictions_lead_{k}.csv", index=False)
        
        # Calculate B-7 Safety metrics
        safety = safety_metrics(y_te, p10, p50, p90, static_conventional)
        print("\n--- Safety & Quantile Metrics ---")
        for k_met, v_met in safety.items():
            print(f"{k_met}: {v_met:.2f}")
            
        # Calculate B-9 Breakdowns
        print("\n--- Performance Breakdowns (Main Pred) ---")
        breakdowns = performance_breakdowns(res_df)
        print("By Wind Bin:\n", breakdowns['WindBin'][['Count', 'MAE']].to_string())

if __name__ == "__main__":
    main()
