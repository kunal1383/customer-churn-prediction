import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.exception import CustomException
from src.logger import logging

import datetime

       
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    
    except Exception as e:

        raise CustomException(e, sys)

def evaluate_xgb_model(X_train, y_train, X_test, y_test, xgb_model):
    try:
        logging.info("Model evalution started")
        # Make predictions on training and test data
        y_train_pred = xgb_model.predict(X_train)
        y_test_pred = xgb_model.predict(X_test)

        # Calculate accuracy, precision, recall, and F1-score for training data
        accuracy_train = accuracy_score(y_train, y_train_pred)
        precision_train = precision_score(y_train, y_train_pred)
        recall_train = recall_score(y_train, y_train_pred)
        f1_train = f1_score(y_train, y_train_pred)

        # Calculate accuracy, precision, recall, and F1-score for test data
        accuracy_test = accuracy_score(y_test, y_test_pred)
        precision_test = precision_score(y_test, y_test_pred)
        recall_test = recall_score(y_test, y_test_pred)
        f1_test = f1_score(y_test, y_test_pred)

        # Create a report with evaluation metrics
        model_report = {
            'Accuracy (Train)': accuracy_train,
            'Precision (Train)': precision_train,
            'Recall (Train)': recall_train,
            'F1-score (Train)': f1_train,
            'Accuracy (Test)': accuracy_test,
            'Precision (Test)': precision_test,
            'Recall (Test)': recall_test,
            'F1-score (Test)': f1_test
        }
        logging.info("Report created")
        return model_report
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)
    
