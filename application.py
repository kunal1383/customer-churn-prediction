import pandas as pd 
import os
import sys
from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.pipeline.training_pipeline import TrainingPipeline
from src.logger import logging
from src.exception import CustomException 


application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/training', methods=['GET', 'POST'])
def training():
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.initiate_training()
        render_template("Training Completed")
    except Exception as e:
        raise CustomException(e ,sys)


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    try:
        if request.method == 'POST':
            # Process form data
            age = int(request.form.get('age'))
            subscription_months = int(request.form.get('Subscription_Length_Months'))
            monthly_bill = float(request.form.get('Monthly_Bill'))
            total_usage_gb = int(request.form.get('Total_Usage_GB'))
            gender = request.form.get('Gender')
            location = request.form.get('Location')
            
            # Convert gender to one-hot encoding (1 for Male, 0 for Female)
            gender_male = 0 if gender == 'Male' else 1

            # Convert location to one-hot encoding (Houston, Los Angeles, Miami, New York)
            location_houston = 1 if location == 'Houston' else 0
            location_los_angeles = 1 if location == 'Los Angeles' else 0
            location_miami = 1 if location == 'Miami' else 0
            location_new_york = 1 if location == 'New York' else 0

            # Create a dictionary with the features
            features = {
                'Age': age,
                'Subscription_Length_Months': subscription_months,
                'Monthly_Bill': monthly_bill,
                'Total_Usage_GB': total_usage_gb,
                'Gender_Male': gender_male,
                'Location_Houston': location_houston,
                'Location_Los_Angeles': location_los_angeles,
                'Location_Miami': location_miami,
                'Location_New_York': location_new_york
            }

            # Convert data to DataFrame
            df = pd.DataFrame([features])

            # Predict using the pipeline
            predict_pipeline = PredictionPipeline()
            pred = predict_pipeline.predict(df)

            if pred == 0:
                result_message = "Customer is not likely to churn."
            else:
                result_message = "Customer is likely to churn."

            return render_template('result.html', final_result=result_message)
    
    except Exception as e:
        raise CustomException(e, sys)
        return render_template('error.html', error_message=str(e))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
