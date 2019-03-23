import pandas as pd
import numpy as np

def readData(filename):
    return pd.read_csv(filename)

dataset = readData("train.csv")

dataSize = dataset.shape[0]

np.random.RandomState(42)
shuffledIndicies = np.random.permutation(np.arrage(dataSize))

testRatio = .2
trainRatio = 1 - testRatio
trainStart = (dataSize * trainRatio) - 1

dataTrain = shuffledIndicies[]
labelsTrain = 