import pickle
import numpy as np
from flask import Flask, request, render_template

# Initialize Flask app
app = Flask(__name__)

# Load the encoders and model
with open('onehot_encoder_weather.pkl', 'rb') as file:
    onehot_encoder_weather = pickle.load(file)

with open('onehot_encoder_traffic.pkl', 'rb') as file:
    onehot_encoder_traffic = pickle.load(file)

with open('label_encoder_delayed.pkl', 'rb') as file:
    label_encoder_delayed = pickle.load(file)

with open('xgboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    shipment_id = int(request.form['shipment_id'])
    origin = request.form['origin']
    destination = request.form['destination']
    shipment_date = request.form['shipment_date']
    planned_delivery_date = request.form['planned_delivery_date']
    vehicle_type = request.form['vehicle_type']
    distance = int(request.form['distance'])
    weather = request.form['weather']
    traffic = request.form['traffic']
    
    # One-hot encode 'Weather Conditions'
    weather_encoded = onehot_encoder_weather.transform([[weather]]).toarray()

    # One-hot encode 'Traffic Conditions'
    traffic_encoded = onehot_encoder_traffic.transform([[traffic]]).toarray()

    # Prepare input data for the model
    input_data = np.concatenate([
        weather_encoded.flatten(),
        traffic_encoded.flatten(),
        [distance]  # Assuming other fields are preprocessed or ignored
    ])

    # Predict if delayed
    prediction = model.predict([input_data])

    # Inverse transform the prediction
    prediction_label = label_encoder_delayed.inverse_transform(prediction)

    return render_template('index.html', prediction=prediction_label[0])

if __name__ == '__main__':
    app.run(debug=True)
