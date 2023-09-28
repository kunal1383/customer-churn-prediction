import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.pipeline.training_pipeline import TrainingPipeline




class PredictionPipeline:
    def __init__(self):
        self.training_pipeline = TrainingPipeline()
        
    def load_models_from_artifacts(self):
        preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        model_path = os.path.join('artifacts', 'model.pkl')

        preprocessor = None
        model = None  
        
        try:
            logging.info("Ckecking if models present in artifacts")
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            logging.info("Model loaded successfully")
            return model ,preprocessor
        except FileNotFoundError:
            logging.info("model not  present in artifacts starting the training of model")
            self.training_pipeline.initiate_training()

        return self.load_models_from_artifacts()
    
    def predict(self, features):
        logging.info('Starting the prediction pipeline')
        try:
            model ,preprocessor = self.load_models_from_artifacts()
            columns_to_scale = ['Age', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']
            features[columns_to_scale] = preprocessor.transform(features[columns_to_scale])
            logging.info(f"Data scaled:{features}")
            prediction = model.predict(features)
            logging.info('Finished the prediction')
            print()
            return int(prediction)
        except Exception as e:
            logging.info("Exception occurred in prediction") 
            return None  
        
        
