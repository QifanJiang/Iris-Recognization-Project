# Iris-Recognization-Project

Our Project Design
----------------------------
1. Localization
    - detect inner boundary and outer boundary of iris by using Hough transform
    - as an outer boundary, choose a circle whose center is closest to the center of the inner boundary
        - in order not to misditect an other circle as a outer boundary

2. Normalization
    - find an coodinate in an original image which corresponds to a point on the new coodinate system for the normalization image(Backward mapping), by getOriginCoord function 
    - implement getOriginCoord for all coordinates on the new coodinate system to create the normalization image
    - if a part of iris is outside of the image, substitute the edge of the image into the new coordinate

3. Enhancement

4. Feature Extraction

5. Matching
    - input feature vectors for train and test data into the IrisMatching function
        - accoding to the given paper, prepare 7 lotated data in total for each train eye
    - implement LinearDiscriminantAnalysis using train dataset to reduce demensions and increase class separability
    - create new feature vectors for train and test data by applying the model
    - calculate smallest values of three similarity measures for each test data

6. Evaluation



----------------------------

Limitation
----------------------------
- We don't use edge detection in the localization process. If we can use it properly, we will be able to improve accuracy.
- It takes much time to load train and test data from database.

----------------------------

Peer evaluation form
----------------------------
Qifan Jiang, qj2172
Though we built and reviewed whole process of this project together, Qifan particularly contributed to the Localization, Enhancement, FeatureExtraction and Evaluation process. When I struggled with detecting outer boundary for a iris, Qifan took the initiative in working on and resolved the problem. 

Yuta Adachi, ya2488


----------------------------