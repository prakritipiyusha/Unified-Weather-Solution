import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    RocCurveDisplay,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

print("main.py is running")

print("Reading past_data_2000-2025.xlsx…")
data = pd.read_excel("past_data_2000-2025.xlsx", parse_dates=["date"], engine="openpyxl")
data.sort_values("date", inplace=True)
data.reset_index(drop=True, inplace=True)

# Binary conversion
for col in ["RainToday", "RainTomorrow"]:
    data[f"{col}_binary"] = data[col].map({"Yes": 1, "No": 0})

# Lag and rolling features with leakage prevention
for lag in [1, 2]:
    data[f"rain_lag_{lag}"] = data["rain"].shift(lag)
    data[f"temp_lag_{lag}"] = data["temperature_2m"].shift(lag)
    data[f"rh_lag_{lag}"] = data["relative_humidity_2m"].shift(lag)

data["rh3_day_avg"] = data["relative_humidity_2m"].shift(1).rolling(3).mean()
data["gust7_max"] = data["wind_gusts_10m"].shift(1).rolling(7).max()
data["month"] = data["date"].dt.month
data["day_of_year"] = data["date"].dt.dayofyear

data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

# Train classification model
rain_yes = data[data["RainTomorrow_binary"] == 1]
rain_no = data[data["RainTomorrow_binary"] == 0].sample(len(rain_yes) * 2, random_state=42)
df_cls = pd.concat([rain_yes, rain_no]).sort_values("date").reset_index(drop=True)

# Time-based split
split_date = df_cls["date"].quantile(0.8)
train_df = df_cls[df_cls["date"] <= split_date]
test_df = df_cls[df_cls["date"] > split_date]

X_train = train_df.drop(columns=["date", "rain", "RainToday", "RainTomorrow", "RainTomorrow_binary"])
y_train = train_df["RainTomorrow_binary"]
X_test = test_df.drop(columns=["date", "rain", "RainToday", "RainTomorrow", "RainTomorrow_binary"])
y_test = test_df["RainTomorrow_binary"]

print("Training classifier…")
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# Feature selection
feat_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": clf.feature_importances_
}).sort_values("importance", ascending=False)
selected_features = feat_df[feat_df["importance"] > 0.01]["feature"].tolist()

X_train = X_train[selected_features]
X_test = X_test[selected_features]
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)[:, 1]
print(classification_report(y_test, preds))
print("ROC AUC:", roc_auc_score(y_test, probs))

sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.show()

fpr, tpr, _ = roc_curve(y_test, probs)
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc_score(y_test, probs)).plot()
plt.show()

# Regression model for rain amount
rain_df = data[data["RainTomorrow_binary"] == 1].copy()
Xr = rain_df[selected_features]
yr = rain_df["rain"]

Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.2, random_state=42)
reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42
)
reg.fit(Xr_tr, yr_tr)

pred_r = reg.predict(Xr_te)
print("MAE:", mean_absolute_error(yr_te, pred_r))
print("RMSE:", np.sqrt(mean_squared_error(yr_te, pred_r)))

plt.scatter(yr_te, pred_r, alpha=0.3)
plt.plot([0, yr_te.max()], [0, yr_te.max()], "--r")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

joblib.dump(clf, "rain_clf.pkl")
joblib.dump(reg, "rain_reg.pkl")
print("Models saved.")

# Forecast on new data
print("Loading forecast…")
recent = pd.read_excel("forecast_New Delhi_openmeteo.xlsx", parse_dates=["date"], engine="openpyxl")
recent.rename(columns={
    "Temperature (°C)": "temperature_2m",
    "Humidity (%)": "relative_humidity_2m",
    "Rain (mm)": "rain",
    "Wind Gusts (km/h)": "wind_gusts_10m",
}, inplace=True)

recent.set_index("date", inplace=True)
daily = pd.DataFrame({
    "rain": recent["rain"].resample("D").sum(),
    "temperature_2m": recent["temperature_2m"].resample("D").mean(),
    "relative_humidity_2m": recent["relative_humidity_2m"].resample("D").mean(),
    "wind_gusts_10m": recent["wind_gusts_10m"].resample("D").mean(),
}).reset_index()

daily["month"] = daily["date"].dt.month
daily["day_of_year"] = daily["date"].dt.dayofyear

for lag in [1, 2]:
    daily[f"rain_lag_{lag}"] = daily["rain"].shift(lag)
    daily[f"temp_lag_{lag}"] = daily["temperature_2m"].shift(lag)
    daily[f"rh_lag_{lag}"] = daily["relative_humidity_2m"].shift(lag)

daily["rh3_day_avg"] = daily["relative_humidity_2m"].shift(1).rolling(3).mean()
daily["gust7_max"] = daily["wind_gusts_10m"].shift(1).rolling(7).max()

daily.dropna(inplace=True)
daily.reset_index(drop=True, inplace=True)

# Ensure all required features exist
for feat in selected_features:
    if feat not in daily.columns:
        daily[feat] = 0

# Automatically adapt to available rows
n_days = min(5, len(daily))
if n_days == 0:
    print("\n No valid forecast rows available after preprocessing.")
else:
    print(f"\nForecast for next {n_days} day(s):")
    for _, row in daily.tail(n_days).iterrows():
        X_row = row[selected_features].to_frame().T
        date = row["date"].strftime("%Y-%m-%d")
        p = clf.predict_proba(X_row)[0, 1]
        amt = reg.predict(X_row)[0] if p > 0.5 else 0.0
        print(f"{date}: {p:.1%} chance → {amt:.2f} mm")

