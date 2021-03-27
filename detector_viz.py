import os
import sys
import cv2

import detector
from utils import load_json
from record import PictureRecorder


class Tester:

    def __init__(self, dataset):
        self.dataset = dataset

        self.pupils = load_json(dataset, 'pupils')
        self.glints = load_json(dataset, 'glints')

        self.title = 'Preview'
        cv2.namedWindow(self.title)
        cv2.createTrackbar('Image', self.title, 0, len(self.glints), self.change_idx)

        self.update(0)

    def change_idx(self, value):
        self.update(value)

    def loop(self):
        cv2.waitKey(0)

    def update(self, idx):
        #img = cv2.imread('inputs/images/' + self.dataset + f'/{idx}.jpg')
        img = cv2.imread(os.path.join('inputs', 'images', self.dataset, f'{idx}.jpg'))
        if img is None:
            return

        c, ax, angle = detector.find_pupil(img)
        gs = detector.find_glints(img, c)

        p = self.pupils[idx]
        cv2.ellipse(img, (int(p['cx']), int(p['cy'])), (int(p['ax']/2), int(p['ay']/2)), int(p['angle']), 0, 360, (255, 0, 0), 2)

        for g in self.glints[idx]:
            cv2.drawMarker(img, (int(g[0]), int(g[1])), (255, 0, 0), cv2.MARKER_CROSS, 15, 2)


        cv2.ellipse(img, (int(c[0]), int(c[1])), (int(ax[0]/2), int(ax[1]/2)), angle, 0, 360, (0, 0, 255), 1)

        for g in gs:
            cv2.drawMarker(img, (int(g[0]), int(g[1])), (0, 0, 255), cv2.MARKER_CROSS, 15, 1)

        cv2.imshow(self.title, img)

        k = cv2.waitKey(1)

        if k == ord('s'):
            PictureRecorder('outputs', img)




def main():
    if len(sys.argv) == 1:
        raise ValueError("You must supply data folder as a program argument.")

    dataset = sys.argv[1]

    Tester(dataset).loop()

if __name__ == '__main__':
    main()