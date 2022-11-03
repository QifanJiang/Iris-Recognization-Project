import cv2
import math

# Search the position of pupil and iris
# Return the index of pupil and iris
def Localization(image):
    # Eliminate the noise
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.medianBlur(img_gray, 7)

    # The position (x,y), and the radius of the circle r
    x_in = 0
    y_in = 0
    r_in = 0
    x_out = 0
    y_out = 0
    r_out = 0
    
    # Apply HoughCircle() to segement the pupil
    inner_circles = cv2.HoughCircles(
        img_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=30, param1=100, param2=30, minRadius=20, maxRadius=100)
    x_in = int(inner_circles[0][0][0])
    y_in = int(inner_circles[0][0][1])
    r_in = int(inner_circles[0][0][2])

    # The position of iris is approximately 50 more length on the redius than the pupil
    return (x_in,y_in,r_in), (x_in,y_in,r_in+50)