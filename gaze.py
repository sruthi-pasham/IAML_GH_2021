import numpy as np
import detector
import cv2
from sklearn.linear_model import LinearRegression


class GazeModel:

    def __init__(self, calibration_images, calibration_positions):
        self.images = calibration_images
        self.positions = calibration_positions
        self.calibrate()

    def calibrate(self):
        #create an empty numpy array
        Dx = np.empty((0,3), float)
        Dy = np.empty((0,3), float)
        
        #enumerate
        for i,im in enumerate(self.images):
            #detect pupil in form ((x, y), (a, b), angle)
            el = detector.find_pupil(im, debug=False)
            #print(el)
            center = el[0]
            #x co-oridnate of center
            cx = center[0]
            #y co-oridnate of center
            cy = center[1]            
            #stack values to form design matrix of form [x,y,1]
            Dx = np.vstack( (Dx, [cx, cy, 1]) )
            Dy = np.vstack( (Dy, [cx, cy, 1]) )
            #print(self.positions[i], '\t',center)
        #print(Dx)
        #print(Dy)
        self.Dx = Dx
        self.Dy = Dy
        #fit the values to regression modesl
        #from pupil centers to the positions from .json file
        self.lrx = LinearRegression().fit(self.Dx, [ x[0] for x in self.positions])
        self.lry = LinearRegression().fit(self.Dy, [ x[1] for x in self.positions])

    def estimate(self, image):
        # cv2.imshow('a', cv2.imread(image))
        # cv2.waitKey()
        #detect pupil in form ((x, y), (a, b), angle)
        el = detector.find_pupil(image, debug=False)
        center = el[0]
        input = [[center[0], center[1], 1]]
        #print(input)
        #return predicted x and y co-ordinates
        return self.lrx.predict(input)[0], self.lry.predict(input)[0]
