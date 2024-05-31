from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    data = request.form.to_dict()
    input_data = pd.DataFrame([data])
    
    # Debugging: print input data
    print("Input data before conversion:", input_data)

    # Convert input data to numeric values
    input_data = input_data.apply(pd.to_numeric, errors='coerce')

    # Debugging: print input data after conversion
    print("Input data after conversion to numeric:", input_data)

    # Ensure the columns are in the correct order and add missing columns with NaN
    expected_columns = ['radius_mean', 'texture_mean', 'radius_worst', 'texture_worst', 'symmetry_mean', 'symmetry_worst']
    input_data = input_data.reindex(columns=expected_columns, fill_value=0)

    # Debugging: print input data after reindexing
    print("Input data after reindexing:", input_data)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Debugging: print scaled input data
    print("Scaled input data:", input_data_scaled)

    # Make prediction
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)[:, 1]

    # Prepare the response
    result = {
        'prediction': 'Malignant' if int(prediction[0]) == 1 else 'Benign',
        'prediction_proba': float(prediction_proba[0])
    }

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
