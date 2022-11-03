import cv2
import numpy as np

# Enhanced the normalized graph
# return the enhanced normalized graph
def Enhancement(image):
    image_reshaped = image.astype(np.uint8)
    image_normalized = cv2.equalizeHist(image_reshaped)
    return image_normalized