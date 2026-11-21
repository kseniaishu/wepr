from flask import Flask, request, render_template
from weather_prediction import get_predictions  # Import your weather prediction code

app = Flask(__name__)

# Route for displaying the input form
@app.route('/')
def index():
    return render_template('index.html')  # Your HTML form file (created above)

# Route for handling the form submission
@app.route('/predict', methods=['POST'])
def predict_weather():
    # Get the user input from the form
    date = request.form['date']
    region = request.form['region']

    # Call the weather prediction function from your weather_prediction.py
    result = get_predictions(date)  # Use the date from the form

    # Generate a response to display the prediction
    prediction_text = f"Weather Prediction for {region} on {date}:"
    prediction_text += f"\nTemperature: {result['temperature']}°C"
    prediction_text += f"\nPrecipitation: {result['precipitation']} cm"

    # Return the prediction to the user
    return render_template('result.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
# Prophet forecasting
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

def get_predictions(date):
    future_date = pd.to_datetime(date)
    future_df = pd.DataFrame({'ds': [future_date]})

    forecast_tavg = prophet_tavg.predict(future_df)
    forecast_prcp = prophet_prcp.predict(future_df)

    temperature = forecast_tavg['yhat'].values[0]
    precipitation = forecast_prcp['yhat'].values[0]

    return {
        'temperature': round(temperature, 2),
        'precipitation': round(precipitation, 2)
    }
