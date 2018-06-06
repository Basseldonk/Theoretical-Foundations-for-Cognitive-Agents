#!/usr/bin/env python3.6

import numpy as np
import cv2

class dataset_generator:
    """
    Shows users an image and lets them indicate whether they like this or not. Feedback is stored to generate a dataset.
    """

    def __init__(self):
        pass

    def showImage(self, img):
        img = cv2.imread(img, 1)
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    dg = dataset_generator
