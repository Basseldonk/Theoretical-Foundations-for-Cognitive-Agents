#!/usr/bin/env python3.6

import numpy as np
import cv2

class dataset_generator:
    """
    Shows users an image and lets them indicate whether they like this or not. Feedback is stored to generate a dataset.
    """

    def __init__(self):
        self.dataset = []
        self.datapath = "./resouces/"
        pass

    def showImage(self, img):
        img = cv2.imread(img, 1)
        cv2.putText(img, "Do you find this person attractive? (Y/N)", (0, 50), 4, (0, 0, 0))
        cv2.imshow("Image", img)
        responded = False
        while ~responded:
            key = cv2.waitKey(0)
            if key == 'Y':
                # TODO store positive value
                cv2.destroyAllWindows()
                responded = True
            if key == 'N':
                # TODO store negative value
                cv2.destroyAllWindows()
                responded = True

if __name__ == "__main__":
    dg = dataset_generator
