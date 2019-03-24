import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attributes=[]):
        self.attributes = attributes
        return self
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attributes].values

def readData(filename):
    return pd.read_csv(filename)

def dropData(data, toDrop=[]):
    data = data.drop(columns=toDrop)
    return data

def cleanData(data, toFill=[]):
    toFill = {name: data[name].mode for name in toFill}
    data = data.fillna(toFill)
    return data

dataset = readData("train.csv")

dropList = ["Cabin", "Ticket", "Name", "PassengerId"]
fillList = ["Embarked", "Sex"]
dataset = dropData(data=dataset, toDrop=dropList)
dataset = cleanData(data=dataset, toFill=fillList)
numCols = dataset.shape[1]

np.random.seed(42)

#Split the datset into train and test, then into features and labels
dataTrain, dataTest = train_test_split(dataset, test_size=.2)
labelsTrain = dataTrain.iloc[:, 0:0]
dataTrain = dataTrain.iloc[:, 1:]
labelsTest = dataTest.iloc[:, 0:0]
dataTest = dataTest.iloc[:, 1:]