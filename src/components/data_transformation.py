import sys
from dataclasses import dataclasses
import pandas as pd
import numpy as np
import os


@dataclasses
class DataTransformationConfig(object):
    preprocessor_object_file_path = os.path.
