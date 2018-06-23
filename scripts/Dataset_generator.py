#!/usr/bin/env python3.6
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
import Facial_Network as nn
import numpy as np
import random
import pickle
import cv2
import os
import time

class dataset_generator:
    """
    Shows users an image and lets them indicate whether they like this or not. Feedback is stored to generate a dataset.
    """

    def __init__(self, demo=False):
        """
        Starts the dataset generator. Use demo=True to only collect data of 50 samples to shorten runtime.
        """
        self.dataset = []
        self.datapath = "../resources/cfd/CFD Version 2.0.3/CFD 2.0.3 Images/"
        sex = input("To which gender are you sexually attracted? (M/F)")
        print(sex)
        self.rateData(sex, demo=demo)

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

    def rateData(self, sex, demo=False):
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
                if demo:  
                    i += 1

class recursive_feature_elimination:

    def removeFeature(self, mlpInput, n):
        leftOuts = []
        for i in range(0, len(mlpInput)):
            leftOuts.append(mlpInput[i].pop(n))
        return mlpInput, leftOuts

    def restoreFeature(self, mlpInput, leftOuts, n):
        for i in range(0, len(mlpInput)):
            mlpInput[i].insert(n, leftOuts[i])
        return mlpInput

    def RFE(self, MLPinput, MLPlabels, attributes):
        attributes = attributes
        mlpPerformances = []
        for i in range(0, len(attributes)):
            print("Removing ", attributes[i], " from list...")
            newMLPinput, leftOuts = self.removeFeature(MLPinput, i)
            clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(32), random_state=1, verbose=True, max_iter=1000, learning_rate="adaptive")
            scores = cross_val_score(clf, MLPinput, MLPlabels, cv=10)
            mlpPerformances.append(scores.mean())
            self.restoreFeature(newMLPinput, leftOuts, i)
        return mlpPerformances

    def labelPerformances(self, MLPinput, MLPlabels):
        attributes = nn.getColNames()
        performances = self.RFE(MLPinput, MLPlabels, attributes)
        lblPerfs = []
        for i in range(0, len(attributes)):
            lblPerfs.append((attributes[i], performances[i]))
        lblPerfs.sort(key= lambda x: x[1])
        return lblPerfs

    def printPerformances(self, lblPerfs):
        for i in range(0, len(lblPerfs)):
            print("Network performance without %s: %f" % (lblPerfs[i][0], lblPerfs[i][1]))


if __name__ == "__main__":
    name = input("What is your name?")
    script_start_time = time.time()
    # Comment out the next part to skip the rating of faces and load an existing file.
    # dg = dataset_generator()
    # pickle.dump(dg.dataset, open("../resources/saved data/saved_data_" + name + ".p", 'wb'))
    MLPinput, MLPlabels = nn.buildMLPtrainInput(pickle.load( open("../resources/saved data/saved_data_" + name + ".p", 'rb')))
    clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(33), random_state=1, verbose=True, max_iter=1000, learning_rate="adaptive")
    general_scores = cross_val_score(clf, MLPinput, MLPlabels, cv=10)

    RFE = recursive_feature_elimination()
    RFEPerformance = RFE.labelPerformances(MLPinput, MLPlabels)
    print("Standard network preformance: \nAccuracy: %f (+/- %f)" % (general_scores.mean(), general_scores.std() * 2))
    RFE.printPerformances(RFEPerformance)

    print('%0.2f min: Finished running networks' % ((time.time() - script_start_time) / 60))
    #Runtime should be around 2 minutes


    #################################################################
    #PERFORMANCES FOR GEMMA, VALUE INDECIES CORRESPOIND WITH THE ATTRIBUTE AT THE SAME INDEX, E.G. THE FIRST ELEMENT IS RACE
    #STORED HERE TEMPORARILY FOR TESTING/EVALUATION PURPOSES
    # [0.7107799671592775, 0.7181691297208539, 0.7110180623973726, 0.7257799671592775, 0.7180541871921182,
    #  0.7216256157635468, 0.7320935960591133, 0.7080541871921182, 0.7107799671592775, 0.7047208538587849,
    #  0.7007963875205254, 0.7180377668308704, 0.7325615763546798, 0.7185221674876847, 0.7280541871921182,
    #  0.7147208538587849, 0.7211494252873563, 0.7420853858784893, 0.7282922824302135, 0.7116174055829229,
    #  0.7213793103448276, 0.7040147783251232, 0.7225697865353038, 0.708152709359606, 0.7216174055829228,
    #  0.7149507389162562, 0.7350656814449918, 0.7348275862068966, 0.7281609195402299, 0.7244827586206897,
    #  0.7726765188834154, 0.5998440065681444, 0.592824302134647, 0.592824302134647, 0.7621921182266009,
    #  0.7305254515599343, 0.7201724137931034, 0.7368390804597702, 0.7688669950738917, 0.7726765188834154,
    #  0.7726765188834154, 0.7726765188834154, 0.7587438423645321, 0.7726765188834154, 0.7760098522167488,
    #  0.7660098522167487, 0.7551806239737273, 0.7519622331691298, 0.7693431855500821, 0.7626765188834155,
    #  0.7563628899835797, 0.7691050903119868, 0.7726765188834154, 0.7726765188834154, 0.7726765188834154,
    #  0.7726765188834154, 0.7726765188834154, 0.7726765188834154, 0.7726765188834154, 0.7726765188834154,
    #  0.7726765188834154, 0.7726765188834154, 0.7726765188834154, 0.7726765188834154, 0.7726765188834154,
    #  0.7726765188834154]
