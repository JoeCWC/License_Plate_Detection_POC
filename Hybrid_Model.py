import numpy as np
import pandas as pd
import cv2
import imutils
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D
from IPython.display import Image
import argparse
import os

def dist(x1, x2, y1, y2):
    return ((x1-x2)**2+(y1-y2)**2)**0.5

labels=pd.read_excel('test_dataset/labels.xlsx')
labels['ID']=labels['ID'].map(str)
labels


