import numpy as np
from matplotlib import pyplot
import cv2
from random import shuffle
from tqdm import tqdm
import os
TRAIN_DIR ='C:\\Users\\Prakash\\Desktop\\TF\\K1\\train'
TEST_DIR ='C:\\Users\\Prakash\\Desktop\\TF\\K1\\test'
IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic')


def label_img(img):
    l=img.split('.')[-3]
    if(l=='cat'):
        return [1,0]
    elif(l=='dog'):
        return [0,1]

def train_data_creator():
    training_data=[]
    for img in os.listdir(TRAIN_DIR):
        label=label_img(img)
        path=os.path.join(TRAIN_DIR,img)
        print(img.split('.')[-2]) 
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
     
        img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        training_data.append((np.array(img),np.array(label)))
    shuffle(training_data)
    np.save('train_data.npy',training_data)
    return training_data

def test_data_creator():
    testing_data=[]
    for img in os.listdir(TEST_DIR):
        path=os.path.join(TEST_DIR,img)
        img_num=img.split('.')[-2]
        print(img.split('.')[-2]) 
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        testing_data.append((np.array(img),img_num))
    shuffle(testing_data)
    np.save('test_data.npy',testing_data)
    return testing_data


train_data=train_data_creator()
test_data=test_data_creator()



    
