# K-Nearest Neighbors (KNN) for Image Comparison
This repository contains the implementation of the K-Nearest Neighbors (KNN) algorithm to find and compare the similarity of images within a given dataset. The project uses Python and several libraries including OpenCV, NumPy, and Matplotlib.

## Table of Contents
1. Introduction
2. Installation
3. Code Explanation
4. Output

### 1. Introduction
The K-Nearest Neighbors (KNN) algorithm is used for both classification and regression tasks. In this project, KNN is applied to compare an input image with a dataset to find the most similar images. The impact of different values of K (1, 3, and 5) on the results is analyzed, highlighting the importance of data preprocessing and optimal parameter selection.

### 2. Installation
To run this project, you need to have Python installed along with the following libraries:

- OpenCV
- NumPy
- Matplotlib

### 3. Code Explanation

Import Libraries
```
import numpy as np
from matplotlib import pyplot as plt
import cv2
from google.colab.patches import cv2_imshow
```

Read and Display Image
```
img = mpimg.imread(image_path)
plt.imshow(img)
plt.axis('off')
plt.show()
print (img.shape)
```

Resize and Normalize Image
```
image = cv2.imread(image_path)
resized_image = cv2.resize(image, (32, 32))
if resized_image.shape[-1] == 1:
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
cv2.imwrite('resized_image.png', resized_image)
cv2_imshow(resized_image)

num_images = images.shape[0]
height, width, channels = 32, 32, 3
print(images.shape)
plt.imshow (resized_image)
```

Compute Euclidean Distance
```
distances = np.linalg.norm(images - your_image_flat, axis=1)
closest_index = np.argmin(distances)
print("Index of the closest image:", closest_index)
```

Display Results
```
# For 3NN
plt.figure(figsize=(10, 4))
for i, index in enumerate(indices):
    plt.subplot(1, 3, i+1)
    plt.imshow(im[:,:,:,index])
    plt.axis('off')
    plt.title(f"Image {index}")
plt.show()
```
For 3NN and 5NN, use loops to find and display the nearest neighbors

### 4. Output of 3NN
![Screenshot 2024-06-29 133023](https://github.com/muhammadtalha72014/KNN_image_comparison/assets/173653061/632b5454-a0bf-4267-a934-ded5edccdce3)
