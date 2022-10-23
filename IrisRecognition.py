# This file is about the main function,
# which will use all the following sub functions


IrisLocalization()
# Detecting pupil and outer boundary of iris
# You can choose other iris localization methods if they work better

IrisNormalization()
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