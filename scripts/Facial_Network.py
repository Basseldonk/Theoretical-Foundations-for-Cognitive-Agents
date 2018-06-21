
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import sys
import random

from PIL import Image
from os import listdir
from os.path import isfile, join
from sklearn.neural_network import MLPClassifier


# In[3]:


main_path = '../resources/cfd/CFD Version 2.0.3/'
excel_path = main_path + 'CFD 2.0.3 Norming Data and Codebook.xlsx'
pictures_path = main_path + 'CFD 2.0.3 Images/'
cfd = pd.read_excel(excel_path, sheet_name = 0, header = 4, skipRows = 4)
cfd.head()


# In[4]:


def encodeRace(race):
    #Asian = 1, Black = 2, Latin = 3, White = 4, Other = 0
    if(race is 'A'):
        return 1
    elif(race is 'B'):
        return 2
    elif(race is 'L'):
        return 3
    elif(race is 'W'):
        return 4
    else:
        return 0


# In[5]:


def encodeGender(gender):
    #Female = 1, Male = 2
    if(gender is 'F'):
        return 1
    elif(gender is 'M'):
        return 2
    else:
        return 0


# In[6]:


cfdOrdinal = cfd.copy(deep=True)
cfdOrdinal['Race'] = cfdOrdinal['Race'].apply(lambda x: encodeRace(x))
cfdOrdinal['Gender'] = cfdOrdinal['Gender'].apply(lambda y: encodeGender(y))
cfdOrdinal = cfdOrdinal.drop('Suitability', axis = 1)
cfdOrdinal = cfdOrdinal.drop('NumberofRaters', axis = 1)
cfdOrdinal.head()


# In[7]:


def accessPicture(target):
    testPath = pictures_path + "" + target
    allFilesInPath = [f for f in listdir(testPath) if isfile(join(testPath, f))]
    neutralPic = next((i_path for i_path in allFilesInPath if 'N' in i_path), (""))
    return Image.open(testPath + '/' + neutralPic)


# In[9]:


#Takes an array of picture names as input, returns an array of all pixel values
def buildMLPtestInput(pictureNames):
    MLPinput = []
    for i in range(0, len(pictureNames)):
        subject = list(cfdOrdinal.loc[cfdOrdinal["Target"] == pictureNames[i]].values[0])
        del subject[0]
        MLPinput.append(subject)
        print('{} / {} complete.'.format(i+1,len(pictureNames)))
        sys.stdout.flush()
    return MLPinput


# In[10]:


def createRandomTrainSet(trainSet):
    networkIn = []
    for i in range(0, len(trainSet)):
        networkIn.append((trainSet[i],random.choice([True, False])))
    return networkIn


# In[11]:


#Takes an array of picture names as input, returns an array of all pixel values
def buildMLPtrainInput(pictureTuples):
    MLPinput = []
    MLPlabels = []
    for i in range(0, len(pictureTuples)):
        current = pictureTuples[i]
        subject = list(cfdOrdinal.loc[cfdOrdinal["Target"] == current[0]].values[0])
        del subject[0]
        MLPinput.append(subject)
        MLPlabels.append(current[1])
        print('{} / {} complete.'.format(i+1,len(pictureTuples)))
        sys.stdout.flush()
    return MLPinput, MLPlabels

#Should result in array of 12,596,376 values
def imageToFloats(image):
    red = []
    green = []
    blue = []
    for i in range(0, image.size[0]):
        for j in range(0, image.size[1]):
            r,g,b = image.getpixel((i,j))
            red.append(r)
            green.append(g)
            blue.append(b)
    return red + green + blue


# In[42]:


#Remove the 'Target' column prior to calling this function
# def checkMat():
#     for row in range (0, len(checkArray)):
#         for col in range (0, len(checkArray[0])):
#             if(type(checkArray[row][col]) is not np.float64):
#                 return False
#     return True
# checkArray = cfdOrdinal.as_matrix()
# checkMat()


# In[30]:


# races = cfd.Race.unique()
# genders = cfd.Gender.unique()
# print("Races in dataset:")
# for race in races:
#     print(race, ": ", len(cfd.loc[cfd['Race'] == race]))
# print('='*50)

# print("Genders in dataset:")
# for gender in genders:
#     print(gender, ": ", len(cfd.loc[cfd['Gender'] == gender]))
# print('='*50)

