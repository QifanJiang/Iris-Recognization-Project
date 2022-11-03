# Iris-Recognization-Project

Our Project Design
----------------------------
1. Localization
    - detect inner boundary and outer boundary of iris by using Hough transform
    - as an outer boundary, choose a circle whose center is the same with the center of the inner boundary, but with redius +50

2. Normalization
    - find an coodinate in an original image which corresponds to a point on the new coodinate system for the normalization image(Backward mapping), by getOriginCoord function 
    - implement getOriginCoord for all coordinates on the new coodinate system to create the normalization image
    - if a part of iris is outside of the image, substitute the edge of the image into the new coordinate

3. Enhancement
    - use cv2.equalizeHist to enhanced the normalized images for feature extraction

4. Feature Extraction
    - use Gabor filtering 4x4 with two channels on our normalized graph to get the m1, sd1, m2, sd2 illustrated in the paper
    - only filtering the top half of the normalized image to avoid the area other than iris
    - store the feature values of all filters in a vector

5. Matching
    - input feature vectors for train and test data into the IrisMatching function
    - accoding to the given paper, prepare 7 lotated data in total for each train eye
    - implement LinearDiscriminantAnalysis using train dataset to reduce demensions and increase class separability
    - create new feature vectors for train and test data by applying the model
    - calculate smallest values of three similarity measures for each test data

6. Evaluation
    - use the dimensionality of size-1 (107) to get the RCC of 3 different matching methods: L1, L2 and Cosine
    - perfrom matching with different dimensionality with the Cosine similarity measure, plot and store the graph as "plot1.png"
    - calculate the FNMR and FMR with different thresholds mentioned in the paper

----------------------------

Limitation
----------------------------
- We don't use edge detection in the localization process. If we can use it properly, we will be able to improve accuracy.
- We didn't choose the clearest image among 3 images in the training dataset, this might cause bias
- It takes much time to load train and test data from database.

----------------------------

Peer evaluation form
----------------------------
To Qifan Jiang, qj2172
Though we built and reviewed whole process of this project together, Qifan particularly contributed to the Localization, Enhancement, FeatureExtraction and Evaluation process. When I struggled with detecting outer boundary for a iris, Qifan took the initiative in working on and resolved the problem. 

To Yuta Adachi, ya2488
Yuta made a huge contribution on the project spcially in the Localization, Normalization, and Matching process. He offered many help on the organization when I finished and added new sub functions to our project. We succefully achieve our goal, and It's really a nice experience working with Yuta.

----------------------------
