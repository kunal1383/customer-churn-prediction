# Customer Churn Prediction

Customer churn, also known as customer attrition, is when a customer stops doing business with a company. Identifying and understanding the factors that lead to customer churn is crucial for businesses to take proactive measures and retain valuable customers. This project focuses on predicting customer churn using machine learning techniques.

## Overview

The customer churn prediction project involves the following key steps:

1. Data Collection: Gathering historical customer data, including features such as age, subscription length, monthly bill, total usage, gender, and location.

2. Data Preprocessing: Cleaning and preparing the dataset for machine learning. This includes handling missing values and encoding categorical variables.

3. Feature Engineering: Creating new features and encoding categorical variables to improve model performance.

4. Model Selection: Evaluating different machine learning models to identify the best-performing one for customer churn prediction.

5. Model Evaluation: Assessing model performance using various metrics and visualizations.

6. Hyperparameter Tuning: Optimizing hyperparameters for the selected model to improve its predictive accuracy.

7. Threshold Optimization: Finding the optimal probability threshold for classification to balance precision and recall.

8. Deployment: Deploying the final model for use in a web application to predict customer churn.

## Setup

To set up the project environment, follow these steps:

1. Clone the repository to your local machine:

```bash
git clone https://github.com/kunal1383/customer-churn-prediction.git
```

2. Create a Conda virtual environment (if not already created):

```bash
conda create -n venv python=3.8
```

3. Activate the virtual environment:

```bash
conda activate venv
```

4. Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Running the Flask Application

To run the Flask application for customer churn prediction, follow these steps:

1. Make sure you have activated your Conda virtual environment (refer to the [Setup](#setup) section).

2. Navigate to the project root directory.

3. Open a terminal or command prompt.

4. Run the following command to start the Flask application:

```bash
python application.py
```

5. The application will start, and you will see output indicating that the server is running on port 8080.

6. Open a web browser and go to [http://localhost:8080](http://localhost:8080) to access the customer churn prediction web interface.

7. Fill in the required input fields and submit the form to get predictions.

8. The application will display whether the customer is predicted to churn (1) or not (0) based on the input data.

9. You can interact with the application to make predictions for different customer scenarios.

10. To stop the Flask application, press `Ctrl+C` in the terminal where it is running.

## Model Selection and Performance

After evaluating various machine learning models, we found that the XGBoost Classifier performed the best for customer churn prediction. We optimized its hyperparameters and probability threshold to achieve a balanced accuracy, precision, recall, and F1-score. The ROC-AUC score for the XGBoost model is approximately 0.90.

## Conclusion

This project provides a framework for predicting customer churn and offers insights into customer retention strategies. By using the deployed model in production, businesses can identify potential churners and take proactive steps to retain valuable customers.

For any questions or further analysis, please refer to the provided code and documentation.

---

