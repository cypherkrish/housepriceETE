import os
import sys


from src.logger import logging
from src.exception import CustomException

import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation

## Initiate the data ingestion configurtion

@dataclass
class DataIngestionConfig(object):
    train_data_path:str = os.path.join('artifacts', 'trai.csv')    
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'raw.csv')

## Data ingestion class
class DataIngestion(object):
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion is initiated")

        try:
            df = pd.read_csv(os.path.join('notebooks/data', 'gemstone.csv'))
            logging.info("Data set read as a pandas data frame")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok= True)
            df.to_csv(self.ingestion_config.raw_data_path, index= False)

            logging.info("Train test split")

            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info("Data ingestion is completed")

            return (
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path
            )
                                                                   

        except Exception as e:
            logging.info('Exception occured at Data_ingestion stage')
            raise CustomException(e, sys)
        
