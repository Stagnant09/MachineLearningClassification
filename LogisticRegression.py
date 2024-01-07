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
images = [cv2.resize(img, (32, 32)) for img in images]

#Initializing b vectors for each class
b_class_cats = np.zeros(32*32)
b_class_dogs = np.zeros(32*32)
b_class_wild = np.zeros(32*32)

def getClass_Probabilities(image, b_class_cats, b_class_dogs, b_class_wild):
    e_power_cats = Math.pow(2.71, np.dot(image, b_class_cats))
    e_power_dogs = Math.pow(2.71, np.dot(image, b_class_dogs))
    e_power_wild = Math.pow(2.71, np.dot(image, b_class_wild))
    sum = e_power_cats + e_power_dogs + e_power_wild
    pr_cats = e_power_cats / sum
    pr_dogs = e_power_dogs / sum
    pr_wild = e_power_wild / sum
    if pr_cats > pr_dogs and pr_cats > pr_wild:
        return np.vectors.array([1, 0, 0]), np.vectors.array([pr_cats, pr_dogs, pr_wild]) #Cat
    elif pr_dogs > pr_cats and pr_dogs > pr_wild:
        return np.vectors.array([0, 1, 0]), np.vectors.array([pr_cats, pr_dogs, pr_wild]) #Dog
    else:
        return np.vectors.array([0, 0, 1]), np.vectors.array([pr_cats, pr_dogs, pr_wild]) #Wild

def error(images, targets, b_class_cats, b_class_dogs, b_class_wild):

    cross_entropy = 0
    for i in range(len(images)):
        image = images[i]
        target = targets[i]
        pr_res = getClass_Probabilities(image, b_class_cats, b_class_dogs, b_class_wild)[1]
        for j in range(len(pr_res)):
            cross_entropy -= target[j] * Math.log(pr_res[j])

#Applying Logistic Regression
for i in range(30):
    err = error(images, targets, b_class_cats, b_class_dogs, b_class_wild)
    alpha = 0.01
    #Applying regression with cross-entropy
    b_class_cats = b_class_cats + alpha * err
    b_class_dogs = b_class_dogs + alpha * err
    b_class_wild = b_class_wild + alpha * err