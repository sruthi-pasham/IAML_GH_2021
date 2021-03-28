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

        Dx = np.empty((0,3), float)
        Dy = np.empty((0,3), float)
        for i,im in enumerate(self.images):
            el = detector.find_pupil(im, debug=False)
            #print(el)
            #((x, y), (a, b), angle)
            center = el[0]
            cx = center[0]
            cy = center[1]
            Dx = np.vstack( (Dx, [cx, cy, 1]) )
            Dy = np.vstack( (Dy, [cx, cy, 1]) )
            #print(self.positions[i], '\t',center)
        #print(Dx)
        #print(Dy)
        self.Dx = Dx
        self.Dy = Dy
        self.lrx = LinearRegression().fit(self.Dx, [ x[0] for x in self.positions])
        self.lry = LinearRegression().fit(self.Dy, [ x[1] for x in self.positions])

    def estimate(self, image):
        # cv2.imshow('a', cv2.imread(image))
        # cv2.waitKey()
        el = detector.find_pupil(image, debug=False)
        center = el[0]
        input = [[center[0], center[1], 1]]
        print(input)
        return self.lrx.predict(input)[0], self.lry.predict(input)[0]
