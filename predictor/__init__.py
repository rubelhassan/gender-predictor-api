import logging
import os
import numpy as np

from .GenderPredictor import GenderPredictor
from .ClassifierType import ClassifierType


def importData():
    trainFeatures = []

    try:
        currentDir = os.path.dirname(__file__)
        filename = os.path.join(currentDir, '../data/names.txt')
        f = open(os.path.realpath(filename), 'r')
        trainFeatures = f.read().split('\n')
        trainFeatures = [data.split(',') for data in trainFeatures]
        trainFeatures = np.array(trainFeatures)
        f.close()
    except Exception as e:
        logging.exception("Error loading training data.")
        print("Error loading training data.")
    finally:
        return trainFeatures


trainFeatures = importData()
