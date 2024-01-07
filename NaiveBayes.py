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

#Counting colors + calculating probabilities
color_counter = np.zeros(256)
color_probabilities_sum = [[0, 0, 0] for _ in range(256)]

for k in range(len(images)):
    img = images[k]
    for i in range(32):
        for j in range(32):
            color_counter[img[i,j]] += 1
            if targets[k] == [1,0,0]:
                color_probabilities_sum[img[i,j]][0] += 1
            if targets[k] == [0,1,0]:
                color_probabilities_sum[img[i,j]][1] += 1
            if targets[k] == [0,0,1]:
                color_probabilities_sum[img[i,j]][2] += 1

color_probabilities = [[0,0,0] for _ in range(256)] # Pr[Class_i = class_j | ColorX = colorx]
for i in range(256):
    if color_counter[i]!= 0:
        color_probabilities[i] = [color_probabilities_sum[i][0]/color_counter[i], color_probabilities_sum[i][1]/color_counter[i], color_probabilities_sum[i][2]/color_counter[i]]

class_probabilities = [[0,0,0] for _ in range(256)] # Pr[ColorX = colorx | Class_i = class_j]
class_total_count = [0, 0, 0] # [x_i = class_i]
for i in range(256):
    summ = class_probabilities_sum[i][0] + class_probabilities_sum[i][1] + class_probabilities_sum[i][2]
    for j in range(3):
        class_probabilities[i][j] += color_probabilities_sum[i][j]
        class_total_count[j] += color_probabilities_sum[i][j]
        if summ!= 0:
            class_probabilities[i][j] /= summ
        #...

#Calculating prior probabilities Pr[Class_i = class_i]
prior_probabilities = [0,0,0]
for i in range(3):
    prior_probabilities[i] = class_total_count[i] / len(images)

def product_calculator(image, k):  # γινόμενο όλων των P(xi|Ck)
    product = 1
    for i in range(256):
        product *= class_probabilities[i][k]
    return product

#Naive-Bayes Classifier
def naive_bayes_classifier(image):
    maxx = -1
    maxx_class = [0,0,0]
    for i in range(3): # For each class
        if prior_probabilities[i]*product_calculator(image, i) > maxx:
            maxx = prior_probabilities[i]*product_calculator(image, i)
            maxx_class = [0,0,0]
            maxx_class[i] = 1
    return maxx_class

for image in images:
    print(naive_bayes_classifier(image))