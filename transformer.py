import pandas as pd
import os

#TODO: clean data
def readData():
    dataPath = os.path.join(os.getcwd(), "train.csv")
    return pd.read_csv(dataPath)

housingData = readData()
print(housingData.head())