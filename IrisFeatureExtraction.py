import numpy as np
from scipy.ndimage import convolve

# Use M1 shown in the paper
def M1(x,y,f):
    return np.cos(2*np.pi*f*np.sqrt(x**2+y**2))

# Calculation of Gabor filtering
def G(x,y,f,delta_x,delta_y):
    return (1/(2*np.pi*delta_x*delta_y)) * np.exp(-1/2*(x**2/delta_x**2 + y**2/delta_y**2)) * M1(x,y,f)

# Define the spatial filter
def SpatialFilter(size,delta_x,delta_y):
    filter = np.zeros((size,size))
    f = 1/delta_y
    for i in range(size):
        for j in range(size):
            filter[i,j] = G(-np.fix(size/2)+i, -np.fix(size/2)+j, f, delta_x, delta_y)
    return filter

# Extract the features of a normalized image
# return a 1-D list contains all the feature values of a normalized image
def FeatureExtraction(image):
    # Use 4x4 spatial filter
    size = 4
    # Define spatial filter in 2 channels
    channel1 = SpatialFilter(size,3,1.5)
    channel2 = SpatialFilter(size,4,1.5)

    # Our region of interest is the top 1/2 area of the normalized image
    ROI = 32
    image_ROI = image[:ROI,:]
    image_filtered1 = convolve(image_ROI, channel1, mode="wrap")
    image_filtered2 = convolve(image_ROI, channel2, mode="wrap")

    V = []
    # Extract the feature values
    for i in range(ROI//size):
        for j in range(len(image[0])//size):
            m1 = np.mean(np.abs(image_filtered1[i*size:(i+1)*size,j*size:(j+1)*size]))
            sd1= np.mean(np.abs(np.abs(image_filtered1[i*size:(i+1)*size,j*size:(j+1)*size])-m1))
            m2 = np.mean(np.abs(image_filtered2[i*size:(i+1)*size,j*size:(j+1)*size]))
            sd2= np.mean(np.abs(np.abs(image_filtered2[i*size:(i+1)*size,j*size:(j+1)*size])-m2))
            V.append(m1)
            V.append(sd1)
            V.append(m2)
            V.append(sd2)
    return V