# Holtrop-MetaDelta Model

This repository provides the official implementation of the **Holtrop-MetaDelta model** for **ship fuel consumption and CO₂ emission estimation**.

The model is based on the article "Physics-guided data-driven modelling uncovers regional shipping emission patterns and cleaner propulsion pathways", which is not yet published. 

The proposed framework combines **machine learning models** with **physics-based resistance and propulsion formulations**, enabling robust and interpretable inference of ship fuel use and emissions under realistic sea-state and environmental conditions.

---

## ⚙️ Methodology Overview

The workflow consists of three main stages:

1. **Calm Water and Added Resistance Prediction**

   Holtrop program is adopted to estimate the calm water resistance.
   A neural-network-based **MetaDelta** is used for stacking.

3. **Fuel Consumption Estimation**  
   Based on physical relationships between:
   - Ship speed relative to water
   - Total resistance (calm + added)
   - Propulsion power
   - Hourly fuel consumption

4. **Daily CO₂ Emission Calculation**  
   Daily fuel consumption is aggregated and converted into CO₂ emissions using a constant emission factor (default: **3.15 tCO₂ / t fuel**).

---

## 🚀 Quick Start: Inference Pipeline

### 1️⃣ Predict Calm Water and Added Resistance

```python
from inference.radded_predictor import predict_r_added_from_dir
from models.class_calm_water_resistance_estimatoin import *
import pandas as pd

df = pd.read_csv('validation_data_55.csv')

df['R_calm'] = df.apply(Cal_R_calm, axis=1)  # kN

feature_cols = [
    'SOG', 'heading', 'draught',
    'wind_val', 'wind_direction',
    'wave_val', 'wave_direction',
    'stream_val', 'stream_direction'
]

r_added, data_with_radded, scaler = predict_r_added_from_dir(
    df=df,
    model_dir="Modellib",
    feature_cols=feature_cols,
    scaler_name="scaler_inference.pkl",
    device="cpu"
)
```
### 2️⃣ Compute Daily Fuel Consumption and CO₂ Emissions

```python
from inference.daily_fuel import compute_daily_fuel

df_hourly, df_daily = compute_daily_fuel(
    data_with_radded,
    time_col="postime",
    emission_factor=3.15
)
```
### 3️⃣ Merge with MRV Ground Truth

```python
import pandas as pd

df_true = pd.read_csv("MRV_true.csv")

df_daily["date"] = pd.to_datetime(df_daily["date"])
df_true["date"] = pd.to_datetime(df_true["date"])

df_merged = df_true.merge(
    df_daily[["Vessel ID", "date", "Predict_CO2"]],
    on=["Vessel ID", "date"],
    how="left"
)
```

---

### ✅ Results preview

| Vessel ID | date       | fuel_true_day | emission_true_day | emission_predict_day |
|-----------|------------|---------------|-------------------|----------------------|
| 1 | 2021-09-23 | 71.43 | 225.0045 | 216.386105 |
| 1 | 2021-09-29 | 64.52 | 203.2380 | 210.359621 |
| 1 | 2021-10-02 | 62.02 | 195.3630 | 192.851040 |
| 1 | 2022-05-04 | 68.61 | 216.1215 | 210.895311 |
| 1 | 2022-07-10 | 52.42 | 165.1230 | 163.911538 |
| 1 | 2023-03-08 | 61.59 | 194.0085 | 185.930617 |
| 1 | 2023-06-06 | 60.67 | 191.1105 | 190.003571 |
| 1 | 2024-02-24 | 48.96 | 154.2240 | 164.527325 |
| 1 | 2024-03-08 | 63.94 | 201.4110 | 202.149131 |
| 2 | 2021-04-13 | 52.21 | 164.4615 | 167.098672 |
| 2 | 2021-05-26 | 59.96 | 188.8740 | 187.958227 |
| 2 | 2021-07-06 | 99.73 | 314.1495 | 301.666259 |
| 2 | 2021-08-30 | 53.18 | 167.5170 | 166.714255 |
| 2 | 2023-03-19 | 44.93 | 141.5295 | 155.077536 |
| 2 | 2023-04-04 | 50.28 | 158.3820 | 158.221306 |
| 2 | 2023-04-12 | 55.92 | 176.1480 | 176.865683 |
| 2 | 2023-05-06 | 51.99 | 163.7685 | 181.208799 |
| 2 | 2023-07-10 | 57.11 | 179.8965 | 180.418726 |
| 2 | 2023-08-14 | 64.08 | 201.8520 | 213.534135 |
| 2 | 2023-09-04 | 51.49 | 162.1935 | 159.805263 |
| 2 | 2023-10-13 | 53.24 | 167.7060 | 164.445844 |

## Contact

If you have any questions, suggestions, or would like to discuss potential collaboration, please feel free to contact:

**Dr. Liang Zhao**  
📧 Email: liamzhao13@zju.edu.cn

## License

This project is licensed under the **MIT License**.


