# This file is about the main function,
# which will use all the following sub functions

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_laplace

# We need to transform data(image) set so that we can deal with it easily
# Now I'm reading an image data directly
path = 'CASIA Iris Image Database (version 1.0)/001/1'
image_file = os.path.join(path, '001_1_2.bmp')
image = cv2.imread(image_file)
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


incir, outcir = IrisLocalization(img_gray)
# Detecting pupil and outer boundary of iris
# You can choose other iris localization methods if they work better

unwrapImage = IrisNormalization(img_gray, incir, outcir)
# Mapping the iris from Cartesian coordinates to polar coordinates

ImageEnhancement()
# Enhancing the normalized iris

FeatureExtraction()
# Filtering the iris and extracting features

IrisMatching()
# Using Fisher linear discriminant for dimension reduction 
# and nearest center classifier for classification

PerformanceEvaluation()
# Calculating the CRR for the identification mode 
# (CRR for all three measures, i.e., L1, L2, and Cosine similarity, should be >=75% , the higher the better), 
# which will output Table 3 & Fig. 10 (refer to Ma’s paper)
# Calculating ROC curve for verification mode, which will output Table 4 and Fig. 11 
# (using Bootstrap and calculating confidence interval is not required).