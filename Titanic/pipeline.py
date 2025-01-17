import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_curve

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attributes=[], returnValues=True):
        self.attributes = attributes
        self.returnValues = returnValues
        return None
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if self.returnValues:
            return X[self.attributes].values
        else:
            return X[self.attributes]


class MultiLabelEncoder(BaseEstimator, TransformerMixin):
    def __init(self):
        return None
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.apply(LabelEncoder().fit_transform)
        return X.values

def getAttributes(data, includeTypes=None, excludeTypes=None):
    return list(data.select_dtypes(include=includeTypes, exclude=excludeTypes).columns)

def readData(filename):
    return pd.read_csv(filename)

def dropData(data, toDrop=[]):
    data = data.drop(columns=toDrop)
    return data

def cleanData(data, toFill=[]):
    toFill = {name: data[name].mode for name in toFill}
    data = data.fillna(toFill)
    data["Embarked"] = data["Embarked"].apply(str)
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
labelsTrain = dataTrain.iloc[:, 0:1]
dataTrain = dataTrain.iloc[:, 1:]
labelsTest = dataTest.iloc[:, 0:1]
dataTest = dataTest.iloc[:, 1:]

numAttr = getAttributes(dataTrain, excludeTypes="object")
catAttr = getAttributes(dataTrain, includeTypes="object")

numPipeline = Pipeline([
    ('selector', DataFrameSelector(numAttr)),
    ('imputer', SimpleImputer(strategy="median")),
    ('standarization', StandardScaler())
])

catPipeline = Pipeline([
    ('selector', DataFrameSelector(catAttr, False)),
    ('multiEncoder', MultiLabelEncoder())
    #('Encoder', OneHotEncoder(sparse=False, categories="auto"))
])

fullPipeline = FeatureUnion(transformer_list=[
    ('numPipeline', numPipeline),
    ('catPipeline', catPipeline)
])

dataTrainPrepared = fullPipeline.fit_transform(dataTrain)
print(dataTrainPrepared.shape)
dataTestPrepared = fullPipeline.fit_transform(dataTest)
print(dataTestPrepared.shape)

sgd = SGDClassifier(max_iter=200,tol=0.01, random_state=42)
sgd.fit(dataTrainPrepared, np.ravel(labelsTrain))
print(sgd.score(dataTestPrepared, labelsTest))

print(cross_val_score(sgd, dataTrainPrepared, np.ravel(labelsTrain), cv=5, scoring="accuracy"))
predictions = cross_val_predict(sgd, dataTrainPrepared, np.ravel(labelsTrain), cv=5)

print(confusion_matrix(np.ravel(labelsTrain), predictions))
print(precision_score(np.ravel(labelsTrain), predictions))
print(recall_score(np.ravel(labelsTrain), predictions))