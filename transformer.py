import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import os
import numpy as np
import hashlib

#Returns a data frame of the list of attributes passed to it
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attributeNames, returnValues=True):
        self.attributeNames = attributeNames
        self.returnValues = returnValues
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if self.returnValues:
            return X[self.attributeNames].values
        else:
            return X[self.attributeNames]

#Returns the data cleaned of null values, and the labels with it
def cleanData(data, colsToDrop=[], rowsToDrop=[], labelId="SalePrice"):
    #Get rid of columns we don't want
    data = data.drop(columns=colsToDrop)
    #Drop rows that have null values
    data = data.dropna(subset=rowsToDrop)
    #Grab the labels
    labels = data[labelId]
    data = data.drop(columns=labelId)
    return data, labels

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

#Read the data and split it
housingData = readData()
dataTrain, dataTest = splitData(housingData, .20, "Id")

#State which columns we want to drop
colDropList = ["Id", "MiscFeature", "Alley", "FireplaceQu", "PoolQC", "Fence"]
#Get all the columns that aren't numerical
catCols = getAttributes(dataTrain, excludeTypes=["int64", "float64"])
#Only remove the rows with an NaN value in columns that we aren't dropping already
rowDropList = [cat for cat in catCols if not cat in colDropList]
#Get the updated training set with its labels
dataTrain, trainLabels = cleanData(dataTrain, colDropList, rowDropList)
dataTest, testLabels = cleanData(dataTest, colDropList, rowDropList)

#Only get the numerical columns
numAttr = getAttributes(dataTrain, excludeTypes=["object"])
#Only get the categorical columns
catAttr = getAttributes(dataTrain, excludeTypes=["int64", "float64"])

#Steps to create the numerical pipeline
numPipeline = Pipeline([
    ("selector", DataFrameSelector(numAttr)),
    ("imputer", SimpleImputer(strategy="median")),
    ("std_scaler", StandardScaler())
])

catPipeline = Pipeline([
    ("selector", DataFrameSelector(catAttr)),
    ("labelEncoder", OrdinalEncoder()),
    #("oneHotEncoder", OneHotEncoder(sparse=False, categories="auto"))
])

fullPipeline = FeatureUnion(transformer_list=[
    ("numPipeline", numPipeline),
    ("catPipeline", catPipeline)
])

dataTrainCleaned = fullPipeline.fit_transform(dataTrain)
print(dataTrainCleaned.shape)
dataTestCleaned = fullPipeline.fit_transform(dataTest)
print(dataTestCleaned.shape)

linReg = LinearRegression()
linReg.fit(dataTrainCleaned, trainLabels)
print(linReg.score(dataTestCleaned, testLabels))