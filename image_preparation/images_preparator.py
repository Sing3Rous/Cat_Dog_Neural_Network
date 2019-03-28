import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import pickle

IMAGE_SIZE = 20

DIR = "C:/Users/singe/Desktop/training/generated_base"

CATEGORIES = ["cats", "dogs"]

def create_training_data(training_data):
    for category in CATEGORIES:
        path = os.path.join(DIR, category)
        class_num = CATEGORIES.index(category)

        for image in tqdm(os.listdir(path)):
            try:
                image_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
                normalized_image_array = cv2.resize(image_array, (IMAGE_SIZE, IMAGE_SIZE))
                training_data.append([normalized_image_array, class_num])

            except Exception:
                pass

training_data = []
create_training_data(training_data)
random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()