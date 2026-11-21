import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from prophet import Prophet

weather = pd.read_csv("huesca.csv", index_col="DATE")


null_pct = weather.isnull().sum() / weather.shape[0]
valid_columns = weather.columns[null_pct < 0.05]
weather = weather[valid_columns].copy()

weather.columns = weather.columns.str.lower()

weather["tavg"] = (weather["tmax"] + weather["tmin"]) / 2

weather["tavg"] = (weather["tavg"] - 32) * 5 / 9

weather["prcp"] = weather["prcp"] * 2.54

weather.index = pd.to_datetime(weather.index)

weather["tavg_target"] = weather["tavg"].shift(-1)
weather["prcp_target"] = weather["prcp"].shift(-1)

weather = weather.ffill()

rr = MultiOutputRegressor(Ridge(alpha=0.1))

predictors = weather.columns[~weather.columns.isin(["tavg_target", "prcp_target", "name", "station"])]


def backtest(weather, model, predictors, start=3650, step=90):
    all_diffs = []
    all_mae = []

    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i, :]
        test = weather.iloc[i:(i + step), :]

        model.fit(train[predictors], train[["tavg_target", "prcp_target"]])

        preds = model.predict(test[predictors])

        mae_temp = mean_absolute_error(test["tavg_target"], preds[:, 0])
        mae_prcp = mean_absolute_error(test["prcp_target"], preds[:, 1])
        all_mae.append((mae_temp, mae_prcp))

        combined = pd.DataFrame({
            "actual_temp": test["tavg_target"],
            "predicted_temp": preds[:, 0],
            "actual_prcp": test["prcp_target"],
            "predicted_prcp": preds[:, 1]
        })
        combined["diff_temp"] = (combined["predicted_temp"] - combined["actual_temp"]).abs()
        combined["diff_prcp"] = (combined["predicted_prcp"] - combined["actual_prcp"]).abs()
        all_diffs.append(combined)

    return all_diffs, all_mae


diffs, maes = backtest(weather, rr, predictors)


mae_temp, mae_prcp = zip(*maes)
print("Mean Absolute Error for Temperature:", sum(mae_temp) / len(mae_temp))
print("Mean Absolute Error for Precipitation:", sum(mae_prcp) / len(mae_prcp))


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


diffs, maes = backtest(weather, rr, predictors)

mae_temp, mae_prcp = zip(*maes)
print("Updated Mean Absolute Error for Temperature:", sum(mae_temp) / len(mae_temp))
print("Updated Mean Absolute Error for Precipitation:", sum(mae_prcp) / len(mae_prcp))

predictions = pd.concat(diffs)
sorted_predictions = predictions.sort_values("diff_temp", ascending=False)
print(sorted_predictions)

print(predictions["diff_temp"].round().value_counts().sort_index())
print(predictions["diff_prcp"].round().value_counts().sort_index())

# introduced the same thing for precipitation, mae=0.061 inches --> into cm rn --> BECAME 0.1544 + say the line responsible for transformation

# Train final model
rr.fit(weather[predictors], weather[["tavg_target", "prcp_target"]])

# Prophet forecasting
def prepare_prophet_df(weather, col):
    df = weather[[col]].reset_index()
    df.columns = ['ds', 'y']
    return df

tavg_df = prepare_prophet_df(weather, 'tavg')
prcp_df = prepare_prophet_df(weather, 'prcp')

# Train Prophet model for temperature
prophet_tavg = Prophet()
prophet_tavg.fit(tavg_df)

# Train Prophet model for precipitation
prophet_prcp = Prophet()
prophet_prcp.fit(prcp_df)

# Predict future weather
future_dates = pd.date_range(start='2027-01-21', end='2027-01-21', freq='D')
future_df = pd.DataFrame({'ds': future_dates})

forecast_tavg = prophet_tavg.predict(future_df)
forecast_prcp = prophet_prcp.predict(future_df)

print(f"Predicted temperature for 2027-01-21: {forecast_tavg['yhat'].values[0]:.2f}°C")
print(f"Predicted precipitation for 2027-01-21: {forecast_prcp['yhat'].values[0]:.2f} cm")

# Plot monthly average temperature and precipitation
import matplotlib.pyplot as plt

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

#now used To make the predictions more precise, including predicting specific dates, you need to use time series forecasting methods that can handle daily data. One approach is to use ARIMA (AutoRegressive Integrated Moving Average) models for univariate time series forecasting or Prophet, which is a tool developed by Facebook for forecasting time series data.


