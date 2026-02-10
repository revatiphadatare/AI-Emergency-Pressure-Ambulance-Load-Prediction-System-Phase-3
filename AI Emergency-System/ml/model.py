import pandas as pd
from sklearn.linear_model import LinearRegression

model = LinearRegression()

def train(data_path):
    data = pd.read_csv(data_path)
    X = data[['hour', 'emergency_calls']]
    y = data['ambulances_available']
    model.fit(X, y)

def predict_load(city, hour, emergency_calls):
    prediction = float(model.predict([[hour, emergency_calls]])[0])
    if prediction < 10:
        pressure = "HIGH"
    elif prediction < 20:
        pressure = "MEDIUM"
    else:
        pressure = "LOW"

    alert = prediction >= 10
    return {
        "city": city,
        "hour": hour,
        "emergency_calls": emergency_calls,
        "predicted_ambulances": round(prediction, 2),
        "pressure_level": pressure,
        "alert_triggered": alert
    }

# Train the model immediately on load
train("data/emergency_data.csv")
