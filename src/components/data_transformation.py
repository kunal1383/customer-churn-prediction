import os
import sys
import pandas as pd 
import numpy as np

from src.exception import CustomException
from src.logger import logging

from sklearn.preprocessing import MinMaxScaler
from src.utils import save_object


from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        
            logging.info('Read train and test data completed')
            

            logging.info('Obtaining preprocessing object')


            target_column_name = "Churn"
            drop_columns=[target_column_name,'CustomerID', 'Name']

            
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]


            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            ## One hot encoding
            input_feature_train_df = pd.get_dummies(input_feature_train_df, columns=['Gender', 'Location'], drop_first=True)
            input_feature_train_df = input_feature_train_df * 1
            
            input_feature_test_df = pd.get_dummies(input_feature_test_df, columns=['Gender', 'Location'], drop_first=True)
            input_feature_test_df = input_feature_test_df * 1
            
            logging.info(f"test_arr :{input_feature_test_df.head()}")
            ## scaling the numerical values
            columns_to_scale = ['Age', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']

            scaler = MinMaxScaler()

            input_feature_train_df[columns_to_scale] = scaler.fit_transform(input_feature_train_df[columns_to_scale])

            input_feature_test_df[columns_to_scale] = scaler.transform(input_feature_test_df[columns_to_scale])


            train_arr = np.c_[input_feature_train_df, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_df, np.array(target_feature_test_df)]
            

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=scaler

            )
            
            return(
                train_arr,
                test_arr
            )
                 
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)