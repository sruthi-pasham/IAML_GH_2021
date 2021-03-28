import cv2
import numpy as np
from utils import *




def find_pupil(img, debug=True):
    """Detects and returns a single pupil candidate for a given image.

    Returns: A pupil candidate in OpenCV ellipse format.
    """
        
    if isinstance(img, str):
        im = cv2.imread(img)
    else:
        im = img
    #convert color to greyscale
    bw_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    #converting to binary image using thresholding
    thres_val = 43
    ret, thres = cv2.threshold(bw_im, thres_val, 255, cv2.THRESH_BINARY_INV)

    #find contours 
    conts, hierarchy = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(conts) < 1:
        return ((-1.0, -1.0), (0.0, 0.0), 0.0)
    
    #find contour with max area
    pupil = max(conts, key=lambda c: cv2.contourArea(c))
    
    #fit ellipse
    el = cv2.fitEllipse(pupil)
    
    #((x, y), (a, b), angle)
    return el
    
    

def find_glints(img, center, debug=True):
    """Detects and returns up to four glint candidates for a given image.

    Returns: Detected glint positions.
    """
    
    if isinstance(img, str):
        im = cv2.imread(img)
    else:
        im = img

    #convert color to greyscale
    bw_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    #converting to binary image using thresholding
    thres_val = 210
    _, thres = cv2.threshold(bw_im, thres_val, 255, cv2.THRESH_BINARY)

    #find contours 
    conts, _ = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort saved contours based on the distance from centre of pupil, in increasing order
    conts.sort(key=lambda c: dist_tuple(get_center(c), center))
    #select 4 closest
    closest4 = conts[:4]
    
    #centre of closest 4
    closest4 = [ get_center(c) for c in closest4 ]
    
    #return 4 closest 
    return closest4
    
    