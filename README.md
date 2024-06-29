# K-Nearest Neighbors (KNN) for Image Comparison
This repository contains the implementation of the K-Nearest Neighbors (KNN) algorithm to find and compare the similarity of images within a given dataset. The project uses Python and several libraries including OpenCV, NumPy, and Matplotlib.

## Table of Contents
1. Introduction
2. Installation
3. Code Explanation
4. Results

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
```
Read and Display Image
python
Copy code
image_path = 'images/your_image.png'
img = plt.imread(image_path)
plt.imshow(img)
plt.axis('off')
plt.show()
Resize and Normalize Image
python
Copy code
image = cv2.imread(image_path)
resized_image = cv2.resize(image, (32, 32))
Compute Euclidean Distance
python
Copy code
distances = np.linalg.norm(images - resized_image.flatten(), axis=1)
closest_index = np.argmin(distances)
Display Results
python
Copy code
# For 1NN
plt.imshow(images[closest_index])
plt.title('1NN Result')
plt.axis('off')
plt.show()

# For 3NN and 5NN, use loops to find and display the nearest neighbors
Results
1NN: Displays the closest image from the dataset.
3NN: Displays the three closest images.
5NN: Displays the five closest images.
Example Output
