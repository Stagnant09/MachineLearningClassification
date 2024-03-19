import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import torch
import torchvision.models as models

#Loading images
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
images = [cv2.resize(img, (8, 8)) for img in images]

#Convert to grayscale
images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

#Building the initial neural network
model = models.resnet18(pretrained=True)

#Training the network
model.fc = torch.nn.Linear(512, 3)
epochs = 3
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
compiled_model = torch.compile(model, criterion, optimizer, epochs)
for i in range(epochs):
    compiled_model.train(images, targets)

#Results
print("Results:")
for i in range(len(images)):
    img = images[i]
    img = torch.tensor(img)
    pred = compiled_model.predict(img)
    pred = pred.detach().numpy()
    pred = np.argmax(pred)
    print("Image: ", i)
    print("Prediction: ", pred)
    print("Target: ", targets[i])
    print("--------------------------------")
