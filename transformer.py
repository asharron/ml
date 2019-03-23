import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import os
import numpy as np
import hashlib

#Returns a data frame of the list of attributes passed to it
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attributeNames):
        self.attributeNames = attributeNames
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attributeNames].values


def getAttributes(data, includeTypes=None, excludeTypes=None):
    return list(data.select_dtypes(include=includeTypes, exclude=excludeTypes).columns)

#TODO: clean data
def readData():
    dataPath = os.path.join(os.getcwd(), "train.csv")
    return pd.read_csv(dataPath)

def testSetCheck(identifier, testRatio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * testRatio

def splitData(data, testRatio, idCol, hash=hashlib.md5):
    ids = data[idCol]
    inTestSet = ids.apply(lambda id_: testSetCheck(id_, testRatio, hash))
    return data.loc[~inTestSet], data.loc[inTestSet]

housingData = readData()
dataTrain, dataTest = splitData(housingData, .20, 'Id')
numAttr = getAttributes(dataTrain, excludeTypes=["object"])
catAttr = getAttributes(dataTrain, excludeTypes=["int64", "float64"])
print("Number Attributes", numAttr)
print("\n")
print("Categorical Attributes", catAttr)