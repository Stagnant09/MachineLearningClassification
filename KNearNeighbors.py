import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import random

images = []
labels = []
targets = []

def find_K_Nearest_Neighbors(X, T, k): # X = data, T = new image to be added to data
    distances = []
    for i in range(len(X)):
        distances.append(np.linalg.norm(X[i] - T))
    distances = np.array(distances)
    indices = distances.argsort()[:k]
    return indices

def find_class(X, T, k):
    indices = find_K_Nearest_Neighbors(X, T, k)
    class_counting_list = [0, 0, 0]
    for indice in indices:
        class_counting_list[labels[indice]-1] += 1
    return np.argmax(class_counting_list)

def add_Image_To_Dataset(data, labels, image):
    data.append(image)
    labels.append(find_class(data, image, 3))
    return data, labels
    
def error(labels, targets):
    return 1/2 * (np.array(labels) - np.array(targets))**2

def reset():
    x = []
    return x

train_folder_path_cats = r"/afhq/train/cat"
train_folder_path_dogs = r"/afhq/train/dog"
train_folder_path_wild = r"/afhq/train/wild"
images, targets = load_images(train_folder_path_cats, [1,0,0])
res = load_images(train_folder_path_dogs, [0,1,0])
images.extend(res[0])
targets.extend(res[1])
res = load_images(train_folder_path_wild, [0,0,1])
images.extend(res[0])
targets.extend(res[1])

def load_images(folder_path, class_name = None, size=(64, 64)):
    images = []
    image_class = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            img = cv2.resize(img, size)  # Resize image
            images.append(img)
            image_class.append(class_name)
    return images, image_class #targets

#Reducing image size
images = [cv2.resize(img, (32, 32)) for img in images]

for i in range(1,11):
    k = i
    print("K = ", k)
    data = reset()
    data = images[:8]
    labels = reset()
    labels = targets[:8]
    for i in range(8, len(images)):
        data, labels = add_Image_To_Dataset(data, labels, images[i])
    print("Error: ", error(labels, targets))
