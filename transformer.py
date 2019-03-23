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

#Returns a list of attributes based on data type
#Meant to be a helper for DataFrameSelector
def getAttributes(data, includeTypes=None, excludeTypes=None):
    return list(data.select_dtypes(include=includeTypes, exclude=excludeTypes).columns)

#Reads data from a csv
def readData():
    dataPath = os.path.join(os.getcwd(), "train.csv")
    return pd.read_csv(dataPath)

#Checks if the ID is in the Test Set
def testSetCheck(identifier, testRatio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * testRatio

#Splits the data into Train / Test
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