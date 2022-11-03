# This file is about the main function,
# which will use all the following sub functions

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import math
import importlib

import IrisLocalization
import IrisNormalization
import IrisEnhancement
import IrisFeatureExtraction
import IrisMatching
import IrisPerformanceEvaluation

importlib.reload(IrisLocalization)
importlib.reload(IrisNormalization)
importlib.reload(IrisEnhancement)
importlib.reload(IrisFeatureExtraction)
importlib.reload(IrisMatching)
importlib.reload(IrisPerformanceEvaluation)

# Generate the feature vectors
def generateFeatureVector(image, incir, outcir, rotate):
    
    img = IrisNormalization.Normalization(image, incir, outcir, rotate)
    img = IrisEnhancement.Enhancement(img)
    V = IrisFeatureExtraction.FeatureExtraction(img)

    return V

def extractImage(flag):
    # flag: 1 if you want to extract train data, 2 if test data
    images = []
    source = 'CASIA Iris Image Database (version 1.0)/'
    n_eye = 108
    if flag == 1: # train data
        n = 3
    elif flag == 2: # test data
        n = 4

    for i in range(1, n_eye+1): # we need to change 2 -> n_eye!
        for j in range(1, n+1):
            path = source + '%03d' % (i,) + '/' + str(flag) + '/'
            filename = '%03d' % (i,) + '_' + str(flag) + '_' + str(j) + '.bmp'
            image = cv2.imread(path + filename)
            images.append(image)
            
    return images

train_images = extractImage(1)
test_images = extractImage(2)

# Store the feature vectors
train_vecs = []
test_vecs = []

for image in train_images:
    incir, outcir = IrisLocalization.Localization(image)
    # Handle the rotation
    degrees = [-9,-6,-3,0,3,6,9]
    for degree in degrees:
        V = generateFeatureVector(image, incir, outcir, degree)
        train_vecs.append(V)
train_vecs = np.array(train_vecs)

for image in test_images:
    incir, outcir = IrisLocalization.Localization(image)
    V = generateFeatureVector(image, incir, outcir, 0)
    test_vecs.append(V)
test_vecs = np.array(test_vecs)

# Different dimensionality
n_components_list = [20,40,60,80,107]
df_min_list = []
df_minmeasure_list = []

for n_components in n_components_list:
    df_min,df_minmeasure = IrisMatching.Matching(train_vecs, test_vecs,n_components)
    df_min_list.append(df_min)
    df_minmeasure_list.append(df_minmeasure)

# Evaluate and output the result, then store a plot of CRR vs Dimensionality on Cosine similarity
IrisPerformanceEvaluation.PerformanceEvaluation(df_min_list,df_minmeasure_list,n_components_list)