import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import random

train_folder_path_cats = r"KostasEdition/afhq/train/cat"
train_folder_path_dogs = r"KostasEdition/afhq/train/dog"
train_folder_path_wild = r"KostasEdition/afhq/train/wild"
images = load_images(train_folder_path_cats, 0)
images.extend(load_images(train_folder_path_dogs), 1)
images.extend(load_images(train_folder_path_wild), 2)

class LeastSquaresModel:
    def __init__(self):
        self.weights = [x for x in random.sample(range(0, 1024), 1024)]
        self.bias = 1
    def calculate(self, x): #Function y = w*x + b
        return sum([self.weights[i] * x[i] for i in range(len(self.weights))]) + self.bias
    def error(self, x, t): #Function error -> t - y_predicted
        return 1/2 * (t - self.calculate(x))**2
    def fix(self, X, T): #Function to fix the error
        self.weights = (np.array(X).T @ np.array(X))^-1 @ (np.array(X).T @ np.array(T))

def get_array_X(images):
    X = []
    for img in images:
        img_array = np.array(img)
        img_array = img_array.flatten()
        X.append(img_array)
    return X

def get_array_T(image_class):
    T = []
    t = []
    for ic in image_class:
        if ic == 0:
            t = [1, 0, 0]
        if ic == 1:
            t = [0, 1, 0]
        if ic == 2:
            t = [0, 0, 1]
        T.append(t)
    return T

def load_images(folder_path, class_name = None, size=(64, 64)):
    images = []
    image_class = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            img = cv2.resize(img, size)  # Resize image
            images.append(img)
            image_class.append(class_name)
    return images

#Edit images so the become 512x512-dimensional
images = [cv2.resize(img, (32, 32)) for img in images]

#Initializing the model
model = LeastSquaresModel()

#Train
model.fix(get_array_X(images), get_array_T(image_class))
