import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load the dataset
earthquake_data = pd.read_csv('cleaned_data.csv')

# Explore the dataset
print(earthquake_data.head())
print(earthquake_data.info())

# Handle missing values
earthquake_data = earthquake_data.dropna()

# Convert date column to datetime
earthquake_data['time'] = pd.to_datetime(earthquake_data['time'])

# Example: Create a feature for time since the last earthquake
earthquake_data = earthquake_data.sort_values('time')
earthquake_data['time_since_last'] = earthquake_data['time'].diff().dt.total_seconds().fillna(0)

# Normalize numerical features
scaler = StandardScaler()
earthquake_data[['mag', 'latitude', 'longitude', 'depth', 'time_since_last']] = scaler.fit_transform(
    earthquake_data[['mag', 'latitude', 'longitude', 'depth', 'time_since_last']]
)

# Define features and target variables
X = earthquake_data[['latitude', 'longitude', 'depth', 'time_since_last']]
y_magnitude = earthquake_data['mag']

# Split the data for magnitude prediction
X_train, X_test, y_train_magnitude, y_test_magnitude = train_test_split(X, y_magnitude, test_size=0.2, random_state=42)

# Initialize and train the model for magnitude prediction
model_magnitude = RandomForestRegressor(n_estimators=100, random_state=42)
model_magnitude.fit(X_train, y_train_magnitude)

# Make predictions and evaluate the model for magnitude
y_pred_magnitude = model_magnitude.predict(X_test)
mae_magnitude = mean_absolute_error(y_test_magnitude, y_pred_magnitude)
print(f'Mean Absolute Error for Magnitude: {mae_magnitude}')

# Define target variable for latitude prediction
y_latitude = earthquake_data['latitude']

# Split the data for latitude prediction
X_train, X_test, y_train_latitude, y_test_latitude = train_test_split(X, y_latitude, test_size=0.2, random_state=42)

# Initialize and train the model for latitude prediction
model_latitude = RandomForestRegressor(n_estimators=100, random_state=42)
model_latitude.fit(X_train, y_train_latitude)

# Make predictions and evaluate the model for latitude
y_pred_latitude = model_latitude.predict(X_test)
mae_latitude = mean_absolute_error(y_test_latitude, y_pred_latitude)
print(f'Mean Absolute Error for Latitude: {mae_latitude}')

# Define target variable for longitude prediction
y_longitude = earthquake_data['longitude']

# Split the data for longitude prediction
X_train, X_test, y_train_longitude, y_test_longitude = train_test_split(X, y_longitude, test_size=0.2, random_state=42)

# Initialize and train the model for longitude prediction
model_longitude = RandomForestRegressor(n_estimators=100, random_state=42)
model_longitude.fit(X_train, y_train_longitude)

# Make predictions and evaluate the model for longitude
y_pred_longitude = model_longitude.predict(X_test)
mae_longitude = mean_absolute_error(y_test_longitude, y_pred_longitude)
print(f'Mean Absolute Error for Longitude: {mae_longitude}')

# Define target variable for time to next earthquake prediction
y_time_to_next = earthquake_data['time_since_last'].shift(-1).fillna(earthquake_data['time_since_last'].mean())

# Split the data for time to next earthquake prediction
X_train, X_test, y_train_time_to_next, y_test_time_to_next = train_test_split(X, y_time_to_next, test_size=0.2, random_state=42)

# Initialize and train the model for time to next earthquake prediction
model_time_to_next = RandomForestRegressor(n_estimators=100, random_state=42)
model_time_to_next.fit(X_train, y_train_time_to_next)

# Make predictions and evaluate the model for time to next earthquake
y_pred_time_to_next = model_time_to_next.predict(X_test)
mae_time_to_next = mean_absolute_error(y_test_time_to_next, y_pred_time_to_next)
print(f'Mean Absolute Error for Time to Next Earthquake: {mae_time_to_next}')

# Number of future earthquakes to predict
n_future_predictions = 10
predictions = []

# Use the most recent earthquake data for the first prediction
latest_earthquake = earthquake_data.iloc[-1][['latitude', 'longitude', 'depth', 'time_since_last']].values.reshape(1, -1)

for _ in range(n_future_predictions):
    # Predict the magnitude, latitude, longitude, and time to next earthquake
    predicted_magnitude = model_magnitude.predict(latest_earthquake)[0]
    predicted_latitude = model_latitude.predict(latest_earthquake)[0]
    predicted_longitude = model_longitude.predict(latest_earthquake)[0]
    predicted_time_to_next = model_time_to_next.predict(latest_earthquake)[0]

    # Convert predicted time to next earthquake to a date
    predicted_date = earthquake_data['time'].iloc[-1] + pd.to_timedelta(predicted_time_to_next, unit='s')

    # Store the prediction
    predictions.append({
        'date': predicted_date,
        'latitude': predicted_latitude,
        'longitude': predicted_longitude,
        'magnitude': predicted_magnitude
    })

    # Update the latest earthquake data for the next prediction
    # Use the scaler's inverse transform to de-normalize the values for updating the latest earthquake
    latest_earthquake = scaler.inverse_transform(latest_earthquake)
    latest_earthquake = np.array([[predicted_latitude, predicted_longitude, 0, predicted_time_to_next]])
    latest_earthquake = scaler.transform(latest_earthquake)

# Print the predictions
for i, prediction in enumerate(predictions):
    print(f'Prediction {i+1}:')
    print(f"  Date: {prediction['date']}")
    print(f"  Latitude: {prediction['latitude']}")
    print(f"  Longitude: {prediction['longitude']}")
    print(f"  Magnitude: {prediction['magnitude']}")

# Visualize the predictions
plt.scatter(earthquake_data['longitude'], earthquake_data['latitude'], c='blue', label='Historical Earthquakes')

for prediction in predictions:
    plt.scatter(prediction['longitude'], prediction['latitude'], c='red', label='Predicted Earthquake', marker='x')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Earthquake Predictions')
plt.legend()
plt.show()

