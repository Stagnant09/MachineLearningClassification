import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os

path_cats = r'C:\Users\Ανδρέας\Εργασίες & Ebooks\4ο έτος\Machine Learning\PROJECT\afhq\train\cat'
path_dogs = r'C:\Users\Ανδρέας\Εργασίες & Ebooks\4ο έτος\Machine Learning\PROJECT\afhq\train\dog'
path_wild = r'C:\Users\Ανδρέας\Εργασίες & Ebooks\4ο έτος\Machine Learning\PROJECT\afhq\train\wild'
paths = [path_cats, path_dogs, path_wild]

#Find Means of Each Channel
red_mean = []
green_mean = []
blue_mean = []
X = []
X = os.listdir(path_cats) + os.listdir(path_dogs) + os.listdir(path_wild)
X_full = []
for image in X:
    done = False
    while done!= True:
        try:
            image = mpimg.imread(os.path.join(paths[0], image))
            X_full.append(image)
            break
        except:
            pass
        try:
            image = mpimg.imread(os.path.join(paths[1], image))
            X_full.append(image)
            break
        except:
            pass
        try:
            image = mpimg.imread(os.path.join(paths[2], image))
            X_full.append(image)
            break
        except:
            pass
    red_mean.append(np.mean(image[:,:,0]))
    green_mean.append(np.mean(image[:,:,1]))
    blue_mean.append(np.mean(image[:,:,2]))

RedMean = 0
for i in range(len(red_mean)):
    RedMean += red_mean[i]
RedMean = RedMean / len(red_mean)
GreenMean = 0
for i in range(len(green_mean)):
    GreenMean += green_mean[i]
GreenMean = GreenMean / len(green_mean)
BlueMean = 0
for i in range(len(blue_mean)):
    BlueMean += blue_mean[i]
BlueMean = BlueMean / len(blue_mean)

#Remove Means
for image in X_full:
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i,j,0] = image[i,j,0] - RedMean
            image[i,j,1] = image[i,j,1] - GreenMean
            image[i,j,2] = image[i,j,2] - BlueMean

#Find transpose of X and cov(X)           

X_Transpose = X_full.T
covariance = np.dot(X_Transpose, X)/len(X)

#Eigenvalue Decomposition of C
eigenvalues, eigenvectors = np.linalg.eig(covariance)