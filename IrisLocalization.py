import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

'''
Detecting pupil and outer boundary of iris.
You can choose other iris localization methods if they work better.
'''

def IrisLocalization(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # projection profiles to find the center coordinates of a pupil (STEP1)
    x_p, y_p, _ = projectionProfile(img_gray, 160, 140, 320, 280)

    # update the center coordinate of the pupil (STEP2)
    ## create binary image around the pupil
    x_p2, y_p2, img_bi2 = projectionProfile(img_gray, x_p, y_p, 120, 120)

    # update again for accuracy (STEP3)
    x_p3, y_p3, img_bi3 = projectionProfile(img_gray, x_p2, y_p2, 120, 120)

    # calculate radius of inner boundary (pupil)
    minx = x_p3 - x_p2 + 60
    miny = y_p3 - y_p2 + 60

    mask = np.where(img_bi3>0, 1, 0)

    # let radius be the average of predicted radius along vertical and horizontal directions
    radius_x = (120 - sum(mask[minx])) / 2
    radius_y = (120 - sum(mask[miny])) / 2
    radius = int((radius_x + radius_y) / 2)


    # detect inner boundary (pupil) using Hough Transform

    # narrow down the image not to detect wrong circles
    img_120 = img_gray[y_p3-60:y_p3+60, x_p3-60:x_p3+60]
    _,img_bi = cv2.threshold(img_120, 127, 255, cv2.THRESH_BINARY)

    # use edge image so that detect iris correctly
    img_edge = gaussian_laplace(img_bi, sigma=3)

    # apply Hough Transform
    for i in range(1,4):
        circles_in = cv2.HoughCircles(img_edge,cv2.HOUGH_GRADIENT,1,100,
                                param1=50,param2=10,minRadius=radius-i,maxRadius=radius+i)

    circles_in = np.uint16(np.around(circles_in[0]))


    # detect outer boundary using Hough transform

    # narrow down the image not to detect wrong circles
    img_260 = img_gray[y_p3-130:y_p3+130, x_p3-130:x_p3+130]
    _,img_bi2 = cv2.threshold(img_260, 127, 255, cv2.THRESH_BINARY)

    # use edge image so that detect iris correctly
    img_edge2 = gaussian_laplace(img_bi2, sigma=3)

    # apply Hough Transform
    for i in range(1,4):
        circles_out = cv2.HoughCircles(img_edge2,cv2.HOUGH_GRADIENT,1,100,
                                param1=30,param2=30,minRadius=70,maxRadius=120)

    circles_out = np.uint16(np.around(circles_out[0]))
    
  

# define a function for projection profile because of repeatitive use
def projectionProfile(image, xp, yp, width, height):
    '''
    image: target image
    xp, yp: center point used when trimming the target image 
    width, height: width and height in which we want to trimming the target image
    '''
    w = int(width/2)
    h = int(height/2)
    # trim the target image 
    img = image[yp-h:yp+h, xp-w:xp+w]

    # create the binary image
    _,img_binary = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)

    # implement projection profile 
    vertical = img_binary.sum(axis=0)
    horizontal = img_binary.sum(axis=1)

    # find a coordinate whose procection profiles are minima
    x_p = np.argmin(vertical) + xp - w
    y_p = np.argmin(horizontal) + yp - h

    return x_p, y_p, img_binary

