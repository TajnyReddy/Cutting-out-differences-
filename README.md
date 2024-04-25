# Cutting out differences with OpenCV
This repository contains Python code for performing image processing tasks using manually applied OpenCV functions.
It requires an original image and a changed image.
![image](https://github.com/TajnyReddy/Cutting-out-differences-/assets/59600478/752b2ad7-4361-4de4-8139-83cc3baa35a4)
![image](https://github.com/TajnyReddy/Cutting-out-differences-/assets/59600478/facec62b-5828-4421-b79e-916022c1c923)

### Overview
The script main.py performs the following tasks:
* Image Difference Calculation: Computes the absolute difference between two input images.
* Thresholding: Converts the difference image to a binary image using a specified threshold value.
* Erosion: Performs erosion operation on the binary image to remove noise and smoothen the edges.
* Connected Components Labeling: Labels connected components in the binary image.
* Bounding Box Detection: Finds the bounding box for each connected component and draws rectangles around them.
* Cutting out the biggest difference: Finds the biggest difference, cuts out the background and saves it.
