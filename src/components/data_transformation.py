import sys
from dataclasses import dataclasses
import pandas as pd
import numpy as np
import os

from src.logger import logging
from src.exception import CustomException


from sklearn.impute import SimpleImputer # Handling missing values
from sklearn.preprocessing import StandardScaler # fro Feature scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

@dataclasses
class DataTransformationConfig(object):
    preprocessor_object_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation(object):
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data transformation is started")

            numerical_cols  = ['carat', 'depth', 'table', 'x', 'y', 'z', 'price']
            categorical_cols = ['cut', 'color', 'clarity']

            cut_categories =  ['Premium', 'Very Good', 'Ideal', 'Good', 'Fair']
            color_categories =  ['F', 'J', 'G', 'E', 'D', 'H', 'I']
            clarity_categories =  ['VS2', 'SI2', 'VS1', 'SI1', 'IF', 'VVS2', 'VVS1', 'I1']

            num_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scalar', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('OrdinalEncoder', OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('scalar', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)]
            )

            return preprocessor

            logging.info("Preprocessing is completed")


        except Exception as e:
            logging.info("Error getting data transformation")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Starting data transformation")
            logging.info("Reading the dta from the train and test paths")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading the train and test data is completed")
            
            logging.info(f'Traind data top 5 records: \n{train_df.head().to_string()}')
            logging.info(f'Testd data top 5 records: \n {test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_object = self.get_data_transformation_object()

            target_column = 'price'
            drop_columns = [target_column, 'id']

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column]

            # Transformation using the preprocessor object
            input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_object.transform(input_feature_test_df)

            logging.info("Preprocessing the train and test data")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path = self.data_transformation_config.preprocessor_object_file_path,
                obj = preprocessing_object

            )

            logging.info("Preprocessor pickle file saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_object_file_path
            )

        except Exception as e:
            logging.info("Error while initiating the data transformation")
            raise CustomException(e, sys)