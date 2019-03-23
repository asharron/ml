import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
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

#Drops and fills missing data
class CleanCategories(BaseEstimator, TransformerMixin):
    def __init__(self, catAttributes):
        self.catAttributes = catAttributes
        return
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.dropna(subset=self.catAttributes)
        return X


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
#TODO: Streamline and refactor dropping of ID column
dataTrain, dataTest = splitData(housingData, .20, "Id")


numAttr = getAttributes(dataTrain, excludeTypes=["object"])

dropList = ["MiscFeature", "Alley", "FireplaceQu", "PoolQC", "Fence"]
categoryList = getAttributes(dataTrain, excludeTypes=["int64", "float64"])
catAttr = [cat for cat in categoryList if not cat in dropList]

dataLabels = CleanCategories(catAttr).fit_transform(dataTrain)['SalePrice']
dataTestLabels = CleanCategories(catAttr).fit_transform(dataTest)['SalePrice']
dataTrain = dataTrain.drop(['Id', 'SalePrice'], axis=1)
dataTest = dataTest.drop(['Id', 'SalePrice'], axis=1)

numAttr = getAttributes(dataTrain, excludeTypes=["object"])
categoryList = getAttributes(dataTrain, excludeTypes=["int64", "float64"])
catAttr = [cat for cat in categoryList if not cat in dropList]

#Steps to create the numerical pipeline
numPipeline = Pipeline([
    ("cleaner", CleanCategories(catAttr)),
    ("selector", DataFrameSelector(numAttr)),
    ("imputer", SimpleImputer(strategy="median")),
    ("std_scaler", StandardScaler())
])

catPipeline = Pipeline([
    ("cleaner", CleanCategories(catAttr)),
    ("selector", DataFrameSelector(catAttr)),
    ("ordEncoder", OrdinalEncoder()),
    ("oneHotEncoder", OneHotEncoder(sparse=False, categories="auto"))
])

fullPipeline = FeatureUnion(transformer_list=[
    ("numPipeline", numPipeline),
    ("catPipeline", catPipeline)
])


dataCleaned = fullPipeline.fit_transform(dataTrain)
linReg = LinearRegression()
linReg.fit(dataCleaned, dataLabels)
print(linReg.score(dataCleaned, dataLabels))