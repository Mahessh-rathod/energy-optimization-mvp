import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DATA_URL = "https://raw.githubusercontent.com/soumilshah1995/Data-Analysis-Over-10-years-of-hourly-energy-consumption-data-from-PJM-in-Megawatts/master/AEP_hourly.csv"

# Optimization simulation settings
PEAK_START, PEAK_END = 14, 20          # hours 14 to 20
OFFPEAK_START, OFFPEAK_END = 1, 5      # hours 1 to 5
SHAVE_PCT = 0.12                       # cut 12% of load in peak hours
SHIFT_SHARE = 0.50                     # move half of the cut energy to off-peak hours


def load_energy_data():
    """Loads the PJM AEP hourly dataset, with a realistic synthetic fallback if offline."""
    try:
        df = pd.read_csv(DATA_URL)
        df.columns = ["Datetime", "Energy_MW"]
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.sort_values("Datetime").drop_duplicates("Datetime").set_index("Datetime")
        df = df.asfreq("h").ffill()
        return df.tail(17520), "PJM AEP hourly (GitHub)"   # last 2 years
    except Exception as e:
        print(f"Network fallback triggered ({e}). Generating synthetic hourly load...")
        idx = pd.date_range(end="2026-11-17 23:00:00", periods=17520, freq="h")
        hours = idx.hour.values
        doy = idx.dayofyear.values
        dow = idx.dayofweek.values
        daily = 2200 * np.sin((hours - 8) * np.pi / 12)
        seasonal = 1800 * np.cos((doy - 200) * 2 * np.pi / 365)
        weekend = np.where(dow >= 5, -900, 0)
        noise = np.random.default_rng(42).normal(0, 350, len(idx))
        df = pd.DataFrame({"Energy_MW": 14500 + daily + seasonal + weekend + noise}, index=idx)
        df.index.name = "Datetime"
        return df, "Synthetic fallback data"


def create_features(df):
    """Calendar clues plus lag (past value) clues."""
    d = df.copy()
    d["hour"] = d.index.hour
    d["dayofweek"] = d.index.dayofweek
    d["month"] = d.index.month
    d["dayofyear"] = d.index.dayofyear
    d["is_weekend"] = (d["dayofweek"] >= 5).astype(int)
    d["lag_24"] = d["Energy_MW"].shift(24)
    d["lag_48"] = d["Energy_MW"].shift(48)
    d["lag_168"] = d["Energy_MW"].shift(168)
    d["rolling_mean_24"] = d["Energy_MW"].shift(24).rolling(24).mean()
    return d


def train_and_forecast(horizon_hours=240):
    print("1/4 Loading hourly energy consumption data...")
    raw_df, source = load_energy_data()
    print(f"    Source: {source}, rows: {len(raw_df):,}")
    feat_df = create_features(raw_df).dropna()

    features = ["hour", "dayofweek", "month", "dayofyear", "is_weekend",
                "lag_24", "lag_48", "lag_168", "rolling_mean_24"]
    target = "Energy_MW"

    # Validation: hold out the last 10 days
    train_val = feat_df.iloc[:-horizon_hours]
    test_val = feat_df.iloc[-horizon_hours:]
    model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8, random_state=42)
    model.fit(train_val[features], train_val[target])
    val_preds = model.predict(test_val[features])
    mape = mean_absolute_percentage_error(test_val[target], val_preds) * 100
    print(f"2/4 Validation MAPE (10-day holdout): {mape:.2f}%")

    # Refit on all data for the real forecast
    model.fit(feat_df[features], feat_df[target])

    print("3/4 Generating +10-day (240-hour) recursive forecast...")
    history = raw_df.copy()
    last_ts = history.index[-1]
    future_ts = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=horizon_hours, freq="h")

    rows = []
    for ts in future_ts:
        history.loc[ts, "Energy_MW"] = np.nan
        temp = create_features(history.tail(200)).iloc[-1:]
        pred = float(model.predict(temp[features])[0])
        history.loc[ts, "Energy_MW"] = pred
        rows.append({"Datetime": ts, "Predicted_MW": round(pred, 2)})
    fc = pd.DataFrame(rows)

    # Simulated optimization: peak shaving + off-peak load shifting
    fc["Hour"] = fc["Datetime"].dt.hour
    fc["Optimized_MW"] = fc["Predicted_MW"]
    peak_mask = fc["Hour"].between(PEAK_START, PEAK_END)
    offpeak_mask = fc["Hour"].between(OFFPEAK_START, OFFPEAK_END)
    shaved = fc.loc[peak_mask, "Predicted_MW"] * SHAVE_PCT
    fc.loc[peak_mask, "Optimized_MW"] -= shaved
    total_shifted = shaved.sum() * SHIFT_SHARE
    fc.loc[offpeak_mask, "Optimized_MW"] += total_shifted / offpeak_mask.sum()
    fc["Optimized_MW"] = fc["Optimized_MW"].round(2)

    # Save outputs
    raw_df.tail(168).reset_index().to_csv(os.path.join(DATA_DIR, "recent_history.csv"), index=False)
    fc.drop(columns=["Hour"]).to_csv(os.path.join(DATA_DIR, "forecast_10d.csv"), index=False)

    peak_row = fc.loc[fc["Predicted_MW"].idxmax()]
    min_row = fc.loc[fc["Predicted_MW"].idxmin()]
    daily_avg = (
        fc.assign(Date=fc["Datetime"].dt.strftime("%Y-%m-%d"))
        .groupby("Date")["Predicted_MW"].agg(["mean", "max", "min"]).round(1)
        .to_dict(orient="index")
    )
    summary = {
        "data_source": source,
        "last_actual_timestamp": str(raw_df.index[-1]),
        "validation_mape_percent": round(mape, 2),
        "forecast_start": str(fc["Datetime"].min()),
        "forecast_end": str(fc["Datetime"].max()),
        "avg_predicted_mw": round(float(fc["Predicted_MW"].mean()), 2),
        "peak_load_mw": float(peak_row["Predicted_MW"]),
        "peak_load_timestamp": str(peak_row["Datetime"]),
        "lowest_load_mw": float(min_row["Predicted_MW"]),
        "lowest_load_timestamp": str(min_row["Datetime"]),
        "optimized_peak_mw": float(fc["Optimized_MW"].max()),
        "total_baseline_mwh": round(float(fc["Predicted_MW"].sum()), 1),
        "total_optimized_mwh": round(float(fc["Optimized_MW"].sum()), 1),
        "estimated_mwh_saved": round(float(fc["Predicted_MW"].sum() - fc["Optimized_MW"].sum()), 1),
        "assumptions": {
            "peak_hours": f"hours {PEAK_START} to {PEAK_END}",
            "offpeak_hours": f"hours {OFFPEAK_START} to {OFFPEAK_END}",
            "peak_shave_percent": int(SHAVE_PCT * 100),
            "share_of_shaved_energy_shifted_percent": int(SHIFT_SHARE * 100),
            "note": "Savings are a simulation based on these assumptions, not measured savings.",
        },
        "daily_breakdown": daily_avg,
    }
    with open(os.path.join(DATA_DIR, "forecast_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("4/4 Done! Saved forecast_10d.csv, recent_history.csv, forecast_summary.json in data/")
    print(f"    Peak load: {summary['peak_load_mw']} MW at {summary['peak_load_timestamp']}")
    print(f"    Simulated energy saved: {summary['estimated_mwh_saved']} MWh")


if __name__ == "__main__":
    train_and_forecast(horizon_hours=240)
