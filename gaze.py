import numpy as np

import detector


class GazeModel:

    def __init__(self, calibration_images, calibration_positions):
        self.images = calibration_images
        self.positions = calibration_positions
        self.calibrate()

    def calibrate(self):
        ...


    def estimate(self, image):
        return 0, 0