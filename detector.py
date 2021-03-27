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

    bw_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    thres_val = 43
    ret, thres = cv2.threshold(bw_im, thres_val, 255, cv2.THRESH_BINARY_INV)

    conts, hierarchy = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(conts) < 1:
        return ((-1.0, -1.0), (0.0, 0.0), 0.0)
    
    pupil = max(conts, key=lambda c: cv2.contourArea(c))
    
    el = cv2.fitEllipse(pupil)
    
    return el
    
    

def find_glints(img, center, debug=True):
    """Detects and returns up to four glint candidates for a given image.

    Returns: Detected glint positions.
    """
    
    if isinstance(img, str):
        im = cv2.imread(img)
    else:
        im = img

    bw_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    thres_val = 210
    _, thres = cv2.threshold(bw_im, thres_val, 255, cv2.THRESH_BINARY)

    conts, _ = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    conts.sort(key=lambda c: dist_tuple(get_center(c), center))
    closest4 = conts[:4]
    closest4 = [ get_center(c) for c in closest4 ]

    return closest4
    
    