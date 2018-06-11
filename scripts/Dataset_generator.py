#!/usr/bin/env python3.6
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
import Facial_Network as nn
import numpy as np
import random
import cv2
import os

class dataset_generator:
    """
    Shows users an image and lets them indicate whether they like this or not. Feedback is stored to generate a dataset.
    """

    def __init__(self):
        self.dataset = []
        self.datapath = "../resources/cfd/CFD Version 2.0.3/CFD 2.0.3 Images/"
        sex = input("To which gender are you sexually attracted? (M/F)")
        print(sex)
        self.rateData(sex)

    def accessPicture(self, target):
        """
        Returns path to image of the neutral picture in the target folder.
        Arguments:
        target = (str) name of the targetfolder
        Returns:
        (str) path of neutral image in target folder
        """
        testPath = self.datapath + "" + target
        allFilesInPath = [f for f in os.listdir(testPath) if os.path.isfile(os.path.join(testPath, f))]
        neutralPic = next((i_path for i_path in allFilesInPath if 'N' in i_path), (""))
        return testPath + '/' + neutralPic

    def showImage(self, img):
        img = cv2.imread(img, 1)
        img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
        cv2.putText(img, "Do you find this person attractive? (Y/N)", (240, 40), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0))
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", img)
        responded = False
        while not responded:
            key = cv2.waitKey(0)
            if key == 121 or key == 89: # if 'Y' or 'y' was pressed.
                cv2.destroyAllWindows()
                return True
            elif key == 78 or key == 110:  # if 'N' or 'n' was pressed.
                cv2.destroyAllWindows()
                return False

    def rateData(self, sex):
        """ 
        Shows image of preferred sex in random order and stores preference rating. 
        Arguments:
        sex = (char) 'M' if you want to rate men, 'F' if you want to rate women. 
        """
        dirs = [f for f in os.listdir(self.datapath) if os.path.isdir(os.path.join(self.datapath, f))]
        random.shuffle(dirs)
        i = 0
        for dir in dirs:
            if dir[1] == sex.upper() and i < 50:
                self.dataset.append((dir, self.showImage(self.accessPicture(dir))))
                i += 1

if __name__ == "__main__":
    dg = dataset_generator()
    print("Len of dataset = {}".format(len(dg.dataset)))
    MLPinput, MLPlabels = nn.buildMLPtrainInput(dg.dataset)
    clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(33), random_state=1, verbose = True)
    scores = cross_val_score(clf, MLPinput, MLPlabels, cv=10)
    print("Accuracy: %0.1f (+/- %0.1f)" % (scores.mean(), scores.std() * 2))
    
