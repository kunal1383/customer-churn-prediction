from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

import os 
import sys


class TrainingPipeline:
    def __init__(self):
        self.train_arr = None
        self.test_arr = None

    def initiate_training(self):
        try:
            logging.info("Starting Training Pipeline")
            obj = DataIngestion()
            train_data_path, test_data_path = obj.initiate_data_ingestion()
            data_transformation = DataTransformation()
            self.train_arr, self.test_arr = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
            model_trainer = ModelTrainer()
            model_trainer.initiate_model_training(self.train_arr, self.test_arr)
            logging.info("Training Pipeline completed successfully")
        except Exception as e:
            raise CustomException(e, sys)
        
        
if __name__=='__main__':
     train = TrainingPipeline()
     train.initiate_training()       