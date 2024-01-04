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
images = load_images(train_folder_path_cats, 0)
images.extend(load_images(train_folder_path_dogs), 1)
images.extend(load_images(train_folder_path_wild), 2)

#Reducing image size
images = [cv2.resize(img, (32, 32)) for img in images]

#Initializing b vectors for each class
b_class_cats = np.zeros(32*32)
b_class_dogs = np.zeros(32*32)
b_class_wild = np.zeros(32*32)

def getClass(image, b_class_cats, b_class_dogs, b_class_wild):
    e_power_cats = Math.pow(2.71, np.dot(image, b_class_cats))
    e_power_dogs = Math.pow(2.71, np.dot(image, b_class_dogs))
    e_power_wild = Math.pow(2.71, np.dot(image, b_class_wild))
    sum = e_power_cats + e_power_dogs + e_power_wild
    pr_cats = e_power_cats / sum
    pr_dogs = e_power_dogs / sum
    pr_wild = e_power_wild / sum
    if pr_cats > pr_dogs and pr_cats > pr_wild:
        return pr_cats
    elif pr_dogs > pr_cats and pr_dogs > pr_wild:
        return pr_dogs
    else:
        return pr_wild

def error(images, targets, b_class_cats, b_class_dogs, b_class_wild):
# 0 = Cat, 0.5 = Dog, 1 = Wild
    cross_entropy = 0
    for i in range(len(images)):
        image = images[i]
        target = targets[i]
        cross_entropy += -target * Math.log(getClass(image, b_class_cats, b_class_dogs, b_class_wild))

#Training