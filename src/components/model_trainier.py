import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass

import os
import sys

@dataclass
class ModelTrainingConfig(object):
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer(object):
    def __init__(self):
        self.model_trainer_confir = ModelTrainingConfig
        