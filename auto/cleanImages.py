import os
from matplotlib.image import imread
import numpy as np

currentDir = os.path.dirname(os.path.realpath(__file__))

imageFiles = np.array([os.path.join(currentDir, "images", f) for f in os.listdir(os.path.join(currentDir, "images")) if os.path.isfile(os.path.join(currentDir, "images", f))])

sizesDict = {}

for imageName in imageFiles:
    image = imread(imageName)
    if not sizesDict.get(image.shape):
        sizesDict[image.shape] = 1
    else:
        sizesDict[image.shape] = sizesDict[image.shape] + 1

with open("./log.log", "w") as f:
    for k,v in sizesDict.items():
        line = str(k) + "-----" + str(v)
        f.write(line)
        print(k, "-------", v)
