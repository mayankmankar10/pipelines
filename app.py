from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from src.datascience.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)  # Initializing a Flask app

@app.route('/', methods=['GET'])  # Route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train', methods=['GET'])  # Route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!"


@app.route('/predict', methods=['POST', 'GET'])  # Route to handle predictions
def index():
    if request.method == 'POST':
        try:
            # Reading the inputs given by the user
            fixed_acidity = float(request.form['fixed_acidity'])
            volatile_acidity = float(request.form['volatile_acidity'])
            citric_acid = float(request.form['citric_acid'])
            residual_sugar = float(request.form['residual_sugar'])
            chlorides = float(request.form['chlorides'])
            free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
            total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
            density = float(request.form['density'])
            pH = float(request.form['pH'])
            sulphates = float(request.form['sulphates'])
            alcohol = float(request.form['alcohol'])
            wine_type = request.form['wine_type']  # Added wine_type for future use

            # Prepare data for prediction
            data = [
                fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                pH, sulphates, alcohol
            ]
            data = np.array(data).reshape(1, -1)

            # Make prediction
            obj = PredictionPipeline()
            predict = obj.predict(data)

            # Convert prediction to a quality score (assuming predict is between 0 and 10)
            quality_score = float(predict[0])  # Ensure prediction is a float
            quality_percentage = (quality_score / 10) * 100  # Convert to percentage

            # Return JSON response for the frontend
            return jsonify({
                'status': 'success',
                'prediction': quality_score,
                'qualityPercentage': quality_percentage,
                'message': 'Prediction successful!'
            })

        except Exception as e:
            print('The Exception message is:', e)
            return jsonify({
                'status': 'error',
                'message': 'Something went wrong! Please try again.'
            }), 500

    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)