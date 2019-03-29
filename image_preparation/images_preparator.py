import numpy as np
import os
import cv2
from tqdm import tqdm
import random
import pickle

def create_data(dir, categories, imageSize):
    data = []
    for category in categories:
        path = os.path.join(dir, category)
        classNum = categories.index(category)

        for image in tqdm(os.listdir(path)):
            try:
                imageArray = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
                normalizedImageArray = cv2.resize(imageArray, (imageSize, imageSize))
                data.append([normalizedImageArray, classNum])

            except Exception:
                pass

    return data

def make_pickle(data, fileName, dataDir):
    pickleOut = open(dataDir + "\pickles\\" + fileName, "wb")
    pickle.dump(data, pickleOut)
    pickleOut.close()

def process_data(data, imageSize):
    X = []
    y = []
    for features, label in data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, imageSize, imageSize, 1)
    X = X / 255.0
    return X, y

categories = ["cats", "dogs"]
print("Enter directory name with data (it must include 2 folders with names \"cats\" and \"dogs\")\n"
      "e. g. C:\\Users\\SingeRous\\Desktop\\training")
dataDir = input()
print("Enter image size for reshaping")
imageSize = int(input())
trainingData = create_data(dataDir + "\\generated\\", categories, imageSize)
testingData = create_data(dataDir + "\\test\\", categories, imageSize)
random.shuffle(trainingData)
random.shuffle(testingData)

XTrain, yTrain = process_data(trainingData, imageSize)
XTest, yTest = process_data(testingData, imageSize)

make_pickle(XTrain, "XTrain.pickle", dataDir + "\\generated\\")
make_pickle(yTrain, "yTrain.pickle", dataDir + "\\generated\\")
make_pickle(XTest, "XTest.pickle", dataDir + "\\test\\")
make_pickle(yTest, "yTest.pickle", dataDir + "\\test\\")