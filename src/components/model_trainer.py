from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_xgb_model
from xgboost import XGBClassifier


from dataclasses import dataclass
import sys
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting dependent and independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Create an XGBoost classifier
            xgb_classifier = XGBClassifier()

            # Train the XGBoost classifier
            xgb_classifier.fit(X_train, y_train)

            # Evaluate the XGBoost classifier
            xgb_classifier_report = evaluate_xgb_model(X_train, y_train, X_test, y_test, xgb_classifier)
            print(xgb_classifier_report)
            print('\n====================================================================================\n')
            logging.info(f'XGBoost Classifier Report : {xgb_classifier_report}')

            # Save the trained XGBoost classifier
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=xgb_classifier
            )
        except Exception as e:
            raise CustomException(e, sys)