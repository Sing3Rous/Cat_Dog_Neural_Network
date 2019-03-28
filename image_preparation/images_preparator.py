import numpy as np
import os
import cv2
from tqdm import tqdm
import random
import pickle

def create_training_data(training_data, dir, categories, image_size):
    for category in categories:
        path = os.path.join(dir, category)
        class_num = categories.index(category)

        for image in tqdm(os.listdir(path)):
            try:
                image_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
                normalized_image_array = cv2.resize(image_array, (image_size, image_size))
                training_data.append([normalized_image_array, class_num])

            except Exception:
                pass

categories = ["cats", "dogs"]
print("Enter directory name with data (it must include 2 folders with names \"cats\" and \"dogs\")\n"
      "e. g. C:/Users/SingeRous/Desktop/training/generated_base")
dir = input()
print("Enter image size for reshaping")
image_size = int(input())
training_data = []
create_training_data(training_data, dir, categories, image_size)
random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, image_size, image_size, 1)
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()