import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Search closest iris among train dataset
# return two list, one of which stores closest iris and another stores their similarity measures
def IrisMatching(train, test):
    '''
    train: a list of feature vectors for train data
    test: a list of feature vectors for test data  
    '''
    train_X = train
    train_y = np.repeat(range(1,109),3)
    test_X = test
    test_y = np.repeat(range(1,109),4)

    # Implement LDA to improve omputational efficiency and classification accuracy 
    clf = LinearDiscriminantAnalysis()
    clf.fit(train_X, train_y)

    f = clf.transform(train_X)
    f_test = clf.transform(test_X)

    df_min = pd.DataFrame(columns=['L1', 'L2', 'Cosine'])
    df_minmeasure = pd.DataFrame(columns=['L1', 'L2', 'Cosine'])


    # Calculate Similarity Measures 
    for i in range(len(f_test)):

        mind1 = 1000
        mind2 = 1000
        mind3 = 1000

        for j in range(len(f)):
            d1 = np.sum(abs(f_test[i:i+1] - f[j:j+1])) 
            d2 = np.sum((f_test[i:i+1] - f[j:j+1])**2)
            d3 = 1 - np.sum(f_test[i:i+1] * f[j:j+1]) / np.sqrt(np.sum(f_test[i:i+1]**2) * np.sum(f[j:j+1]**2))

            # Three Similarity Measures
            if mind1 > d1:
                mind1 = d1
                closest_d1 = j + 1
            if mind2 > d2:
                mind2 = d2
                closest_d2 = j + 1
            if mind3 > d3:
                mind3 = d3
                closest_d3 = j + 1
            
        # Store outputs
        df_min.loc[i+1] = [closest_d1, closest_d2, closest_d3]
        df_minmeasure.loc[i+1] = [mind1, mind2, mind3]
    
    return df_min, df_minmeasure