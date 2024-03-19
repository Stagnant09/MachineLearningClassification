import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

img = mpimg.imread(r'\afhq\train\cat\flickr_cat_000002.jpg')
print('This image is: ',type(img), 
     'with dimensions:', img.shape)
plt.imshow(img)
plt.show()

#PCA

#Mean Removal

n_image = img.copy()

red_mean = np.mean(n_image[:,:,0])
green_mean = np.mean(n_image[:,:,1])
blue_mean = np.mean(n_image[:,:,2])

for i in range(n_image.shape[0]):
    for j in range(n_image.shape[1]):
        n_image[i,j,0] = n_image[i,j,0] - red_mean
        n_image[i,j,1] = n_image[i,j,1] - green_mean
        n_image[i,j,2] = n_image[i,j,2] - blue_mean

n_transpose = n_image.T
