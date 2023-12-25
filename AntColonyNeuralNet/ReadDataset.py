import pandas as pd
import os
import numpy as np

def read_dataset(path):
    return pd.read_csv(path, delimiter=',')

