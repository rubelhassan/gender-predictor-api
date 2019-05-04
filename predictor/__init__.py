import logging
import os
import numpy as np

from .GenderPredictor import GenderPredictor

try:
    currentDir = os.path.dirname(__file__)
    filename = os.path.join(currentDir, '../data/names.txt')
    f = open(os.path.realpath(filename), 'r')
    names_data = f.read().split('\n')
    names_data = [data.split(',') for data in names_data]
    names_data = np.array(names_data)
    f.close()
except Exception as e:
    logging.exception("Error loading training data.")
    print("Error loading training data.")
