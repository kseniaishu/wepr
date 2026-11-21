import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from prophet import Prophet
import matplotlib.pyplot as plt

# Load and preprocess data (unchanged)
weather = pd.read_csv("weather_madrid.csv", index_col="DATE")
null_pct = weather.isnull().sum() / weather.shape[0]
valid_columns = weather.columns[null_pct < 0.05]
weather = weather[valid_columns].copy()
weather.columns = weather.columns.str.lower()
weather["tavg"] = (weather["tavg"] - 32) * 5 / 9
weather["prcp"] = weather["prcp"] * 2.54
weather.index = pd.to_datetime(weather.index)

# Create targets
weather["tavg_target"] = weather["tavg"].shift(-1)
weather["prcp_target"] = weather["prcp"].shift(-1)
weather = weather.ffill()

# Define predictors
predictors = weather.columns[~weather.columns.isin(["tavg_target", "prcp_target", "name", "station"])]

# Save time by reducing steps
step = 90
start = 3650

# Separate models for temperature and precipitation
rr_temp = Ridge(alpha=0.1)
rf_prcp = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)  # Simplified model

# Optimized backtesting function
def backtest(weather, temp_model, prcp_model, predictors, start=3650, step=90):
    all_diffs = []
    all_mae = []

    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i, :]
        test = weather.iloc[i:(i + step), :]

        temp_model.fit(train[predictors], train["tavg_target"])
        prcp_model.fit(train[predictors], train["prcp_target"])

        temp_preds = temp_model.predict(test[predictors])
        prcp_preds = prcp_model.predict(test[predictors])

        mae_temp = mean_absolute_error(test["tavg_target"], temp_preds)
        mae_prcp = mean_absolute_error(test["prcp_target"], prcp_preds)
        all_mae.append((mae_temp, mae_prcp))

        combined = pd.DataFrame({
            "actual_temp": test["tavg_target"],
            "predicted_temp": temp_preds,
            "actual_prcp": test["prcp_target"],
            "predicted_prcp": prcp_preds
        })
        combined["diff_temp"] = (combined["predicted_temp"] - combined["actual_temp"]).abs()
        combined["diff_prcp"] = (combined["predicted_prcp"] - combined["actual_prcp"]).abs()
        all_diffs.append(combined)

    return all_diffs, all_mae

# Additional computations (unchanged)
def pct_diff(old, new):
    return (new - old) / old

def compute_rolling(weather, horizon, col):
    label = f"rolling_{horizon}_{col}" 
    weather[label] = weather[col].rolling(horizon).mean()
    weather[f"{label}_pct"] = pct_diff(weather[label], weather[col])
    return weather

rolling_horizons = [3, 14]
for horizon in rolling_horizons:
    for col in ["tavg", "prcp"]:
        weather = compute_rolling(weather, horizon, col)

weather = weather.iloc[max(rolling_horizons):, :]
weather = weather.fillna(0)

for col in ["tavg", "prcp"]:
    weather[f"month_avg_{col}"] = weather[col].groupby(weather.index.month, group_keys=False).apply(lambda x: x.expanding(1).mean())
    weather[f"day_avg_{col}"] = weather[col].groupby(weather.index.day_of_year, group_keys=False).apply(lambda x: x.expanding(1).mean())

# Perform backtesting with the simplified model
diffs, maes = backtest(weather, rr_temp, rf_prcp, predictors)
mae_temp, mae_prcp = zip(*maes)
print("Updated Mean Absolute Error for Temperature:", sum(mae_temp) / len(mae_temp))
print("Updated Mean Absolute Error for Precipitation:", sum(mae_prcp) / len(mae_prcp))

predictions = pd.concat(diffs)
sorted_predictions = predictions.sort_values("diff_temp", ascending=False)
print(sorted_predictions)
print(predictions["diff_temp"].round().value_counts().sort_index())
print(predictions["diff_prcp"].round().value_counts().sort_index())

# Train final models
rr_temp.fit(weather[predictors], weather["tavg_target"])
rf_prcp.fit(weather[predictors], weather["prcp_target"])

# Prophet forecasting (unchanged)
def prepare_prophet_df(weather, col):
    df = weather[[col]].reset_index()
    df.columns = ['ds', 'y']
    return df

tavg_df = prepare_prophet_df(weather, 'tavg')
prcp_df = prepare_prophet_df(weather, 'prcp')

prophet_tavg = Prophet()
prophet_tavg.fit(tavg_df)

prophet_prcp = Prophet()
prophet_prcp.fit(prcp_df)

future_dates = pd.date_range(start='2031-06-21', end='2031-06-21', freq='D')
future_df = pd.DataFrame({'ds': future_dates})

forecast_tavg = prophet_tavg.predict(future_df)
forecast_prcp = prophet_prcp.predict(future_df)

print(f"Predicted temperature for 2031-06-21: {forecast_tavg['yhat'].values[0]:.2f}°C")
print(f"Predicted precipitation for 2031-06-21: {forecast_prcp['yhat'].values[0]:.2f} cm")

# Feature importance for precipitation prediction
feature_importance = pd.DataFrame({
    'feature': predictors,
    'importance': rf_prcp.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 important features for precipitation prediction:")
print(feature_importance.head(10))

# Two-step model for precipitation
def rain_occurred(x):
    return 1 if x > 0 else 0

weather['rain_occurred'] = weather['prcp'].apply(rain_occurred)

rf_rain_occurrence = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)  # Simplified model
rf_rain_amount = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)  # Simplified model

rf_rain_occurrence.fit(weather[predictors], weather['rain_occurred'])
rf_rain_amount.fit(weather[weather['rain_occurred'] == 1][predictors], weather[weather['rain_occurred'] == 1]['prcp'])

def predict_precipitation(X):
    rain_prob = rf_rain_occurrence.predict_proba(X)[:, 1]
    rain_amount = rf_rain_amount.predict(X)
    return rain_prob * rain_amount

# Use this function for final precipitation predictions
final_prcp_preds = predict_precipitation(weather[predictors])

print("MAE for two-step precipitation model:", mean_absolute_error(weather['prcp'], final_prcp_preds))

monthly_avg_temp = weather.groupby(weather.index.month)["tavg"].mean()
monthly_avg_prcp = weather.groupby(weather.index.month)["prcp"].mean()

plt.figure(figsize=(10, 6))
plt.plot(monthly_avg_temp.index, monthly_avg_temp, marker='o', linestyle='-')
plt.title('Monthly Average Temperature')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')
plt.xticks(range(1, 13))
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(monthly_avg_prcp.index, monthly_avg_prcp, marker='o', linestyle='-')
plt.title('Monthly Average Precipitation')
plt.xlabel('Month')
plt.ylabel('Precipitation (cm)')
plt.xticks(range(1, 13))
plt.grid(True)
plt.show()

predictions['date'] = weather.iloc[max(3650, step):].index
plt.figure(figsize=(14, 7))

plt.subplot(2, 1, 1)
plt.plot(predictions['date'], predictions['actual_temp'], label='Actual Temperature', color='blue')
plt.plot(predictions['date'], predictions['predicted_temp'], label='Predicted Temperature', color='red', linestyle='--')
plt.title('Temperature Prediction Trends')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(predictions['date'], predictions['actual_prcp'], label='Actual Precipitation', color='blue')
plt.plot(predictions['date'], predictions['predicted_prcp'], label='Predicted Precipitation', color='red', linestyle='--')
plt.title('Precipitation Prediction Trends')
plt.xlabel('Date')
plt.ylabel('Precipitation (cm)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(weather['prcp'], bins=50, alpha=0.5, label='Actual')
plt.hist(final_prcp_preds, bins=50, alpha=0.5, label='Predicted')
plt.title('Distribution of Actual vs Predicted Precipitation')
plt.xlabel('Precipitation (cm)')
plt.ylabel('Frequency')
plt.legend()
plt.show()
