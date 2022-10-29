import cv2
import numpy as np

def IrisEnhancement(image):
    image_reshaped = image.astype(np.uint8)
    image_normalized = cv2.equalizeHist(image_reshaped)
    return image_normalized