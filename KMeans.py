import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import random

#Loading images
train_folder_path_cats = r"KostasEdition/afhq/train/cat"
train_folder_path_dogs = r"KostasEdition/afhq/train/dog"
train_folder_path_wild = r"KostasEdition/afhq/train/wild"
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
images = [cv2.resize(img, (8, 8)) for img in images]

#Convert to grayscale
images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

#Number of means
k = 3

#Initialize means with random numbers in [0,256]
means = np.random.randint(0, 256, (k, 8, 8))
neighbors = np.zeros(len(images))

#Train
for epochs in range(10):
    for i in range(len(images)):
        img = images[i]
        #Calculate distances
        distances = np.zeros(k)
        for j in range(k):
            distances[j] = np.linalg.norm(img - means[j])
        neighbors[i] = argmin(distances)
    #Calculate new means
    for i in range(k):
        means[i] = np.mean(images[neighbors == i], axis=0)

#Test
for i in range(len(images)):
    img = images[i]
    #Calculate distances
    distances = np.zeros(k)
    for j in range(k):
        distances[j] = np.linalg.norm(img - means[j])
    neighbors[i] = argmin(distances)
    if (targets[i] == [1,0,0]) and (neighbors[i] == 0):
        print("Cat detected successfully")
    elif (targets[i] == [0,1,0]) and (neighbors[i] == 1):
        print("Dog detected successfully")
    elif (targets[i] == [0,0,1]) and (neighbors[i] == 2):
        print("Wild detected successfully")
    else:
        print("Error")


    
