import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging