"""
This module is used for getting descriptors of contour-based connected
components.

The main method to use is: get_contour_properties(contour, properties=[]):
contour: the contours variable found by cv2.findContours()
properties: list of strings specifying which properties should be
            calculated and returned.

The following properties can be specified:
Approximation: A contour shape to another shape with less number of vertices.
Area: Area within the contour - float.
Boundingbox: Bounding box around contour - 4 tuple (topleft.x, topleft.y,
                                                    width, height).
Centroid: The center of contour - (x, y).
Circle: The the circumcircle of an object.
Circularity: Used to check if the countour is a circle.
Convexhull: Calculates the convex hull of the contour points.
Extend: Ratio of the area and the area of the bounding box. Expresses how
        spread out the contour is.
Ellipse: Fit an ellipse around the contour.
IsConvex: Boolean value specifying if the set of contour points is convex.
Length: Length of the contour
Moments: Dictionary of moments.
Perimeter: Permiter of the contour - equivalent to the length.
RotatedBox: Rotated rectangle as a Box2D structure.

Returns: Dictionary with key equal to the property name.

Example:
    image, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST,
                                                  cv2.CHAIN_APPROX_SIMPLE)
    goodContours = []
    for contour in contours:
        vals = blobl_properties.get_contour_properties(contour, ["area", "length",
                                                                 "centroid", "extend",
                                                                 "convexHull"])
        if vals["area"] > 100 and vals["area"] < 200:
            goodContours.append(contour)
"""

import cv2
import math
import numpy as np


def calculate_approximation(contour):
    """
    Calculate the approximation of a contour shape to another shape with
    less number of vertices depending upon the precision we specify.
    """
    epsilon = 0.1 * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


def calculate_area(contour):
    """
    Calculate the contour area by the function cv2.contourArea() or
    from moments, M["m00"].
    """
    return cv2.contourArea(contour)


def calculate_bounding_box(contour):
    """
    Calculate the bouding rectangle. It is a straight rectangle, it
    doesn't consider the rotation of the object. So area of the bounding
    rectangle won't be minimum. It is found by the function
    cv2.boundingRect().
    """
    return cv2.boundingRect(contour)


def calculate_centroid(contour):
    """
    Calculates the centroid of the contour. Moments up to the third
    order of a polygon or rasterized shape.
    """
    moments = cv2.moments(contour)

    centroid = (-1, -1)
    if moments["m00"] != 0:
        centroid = (int(round(moments["m10"] / moments["m00"])),
                    int(round(moments["m01"] / moments["m00"])))

    return centroid


def calculate_circle(contour):
    """
    Calculate the circumcircle of an object using the function
    cv2.minEnclosingCircle(). It is a circle which completely covers
    the object with minimum area.
    """
    return cv2.minEnclosingCircle(contour)


def calculate_convex_hull(contour):
    """
    Finds the convex hull of a point set by checking a curve for
    convexity defects and corrects it.
    """
    return cv2.convexHull(contour)


def calculate_ellipse(contour):
    """
    Fit an ellipse to an object. It returns the rotated rectangle
    in which the ellipse is inscribed.
    """
    if len(contour) > 5:
        return cv2.fitEllipse(contour)

    return cv2.minAreaRect(contour)


def calculate_extend(contour):
    """
    Calculate the countour extend.
    """
    area = calculate_area(contour)
    boundingBox = calculate_bounding_box(contour)
    return area / (boundingBox[2] * boundingBox[3])


def is_convex(contour):
    """
    Check if a curve is convex or not.
    """
    return cv2.isContourConvex(contour)


def calculate_length(curve):
    """
    Calculate a contour perimeter or a curve length.
    """
    return cv2.arcLength(curve, True)


def calculate_moments(contour):
    """
    Calculate the contour moments to help you to calculate some features
    like center of mass of the object, area of the object etc.
    """
    return cv2.moments(contour)


def calculate_perimeter(curve):
    """Calculates a contour perimeter or a curve length."""
    return cv2.arcLength(curve, True)


def calculate_rotated_box(contour):
    """
    Calculate the rotated rectangle as a Box2D structure which contains
    following detals: (center(x, y), (width, height), angle of rotation).
    """
    rectangle = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rectangle)
    return np.int0(box)


def get_contour_properties(contour, properties=[]):
    """Calcule and return a list of strings specifying by properties."""
    # Initial variables.
    failInInput  = False
    actions = {
        'approximation': calculate_approximation,
        'area': calculate_area,
        'boundingbox': calculate_bounding_box,
        'centroid': calculate_centroid,
        'circle': calculate_circle,
        'convexhull': calculate_convex_hull,
        'extend': calculate_extend,
        'ellipse': calculate_ellipse,
        'isconvex': is_convex,
        'length': calculate_length,
        'moments': calculate_moments,
        'perimeter': calculate_perimeter,
    }
    props = {k: actions[k.lower()] for k in properties}

    return props