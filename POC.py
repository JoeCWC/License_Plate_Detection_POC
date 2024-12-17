import os
import numpy as np
import pandas as pd
import cv2
import imutils
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import easyocr
import glob

from sklearn.metrics import f1_score
import tensorflow as tf
import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Input, Dropout
from tensorflow.python.keras.models import Model, Sequential
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizer_v2.adam import Adam

def import_image(input_file):
    input_file = "/Users/Joe/Documents/Uni_Taipei/Image_Process/License_Plate_Detection/material/test.jpg"
    if os.path.isfile(input_file):
        print(f"The file '{input_file}' exists.")
        image = cv2.imread(input_file)
        return image
    else:
        print("[ERROR]", f"The file '{input_file}' does not exist.")
        exit()

def find_contours(image):
    # Resize image
    image = imutils.resize(image, width=500)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the original image
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    ax[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[0, 0].set_title('Original Image')

    # RGB to Gray scale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ax[0, 1].imshow(gray, cmap='gray')
    ax[0, 1].set_title('Grayscale Conversion')

    # Noise removal with iterative bilateral filter(removes noise while preserving edges)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    ax[1, 0].imshow(gray, cmap='gray')
    ax[1, 0].set_title('Bilateral Filter')

    # Find Edges of the grayscale image
    edged = cv2.Canny(gray, 170, 200)
    ax[1, 1].imshow(edged, cmap='gray')
    ax[1, 1].set_title('Canny Edges')

    fig.tight_layout()
    plt.show()

    # Find contours based on Edges
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[
           :30]  # sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
    NumberPlateCnt = None  # we currently have no Number plate contour

    # loop over our contours to find the best possible approximate contour of number plate
    count = 0
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  # Select the contour with 4 corners
            NumberPlateCnt = approx  # This is our approx Number Plate Contour
            x, y, w, h = cv2.boundingRect(c)
            ROI = img[y:y + h, x:x + w]
            break

    if NumberPlateCnt is not None:
        # Drawing the selected contour on the original image
        cv2.drawContours(image, [NumberPlateCnt], -1, (0, 255, 0), 3)
    print(NumberPlateCnt)

    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    ax[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[0, 0].set_title("Detected license plate")
    # Find bounding box and extract ROI (Region of Interest)
    ax[0, 1].imshow(ROI)
    ax[0, 1].set_title("Extracted license plate")


input_file="/Users/Joe/Documents/Uni_Taipei/Image_Process/License_Plate_Detection/material/test.jpg"
if os.path.isfile(input_file):
    print(f"The file '{input_file}' exists.")
else:
    print("[ERROR]",f"The file '{input_file}' does not exist.")
    exit()
image = cv2.imread(input_file)
# Resize image
image = imutils.resize(image, width=500)
img=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the original image
fig, ax = plt.subplots(2, 2, figsize=(10,7))
ax[0,0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax[0,0].set_title('Original Image')

# RGB to Gray scale conversion
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ax[0,1].imshow(gray, cmap='gray')
ax[0,1].set_title('Grayscale Conversion')

# Noise removal with iterative bilateral filter(removes noise while preserving edges)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
ax[1,0].imshow(gray, cmap='gray')
ax[1,0].set_title('Bilateral Filter')

# Find Edges of the grayscale image
edged = cv2.Canny(gray, 170, 200)
ax[1,1].imshow(edged, cmap='gray')
ax[1,1].set_title('Canny Edges')

fig.tight_layout()
plt.show()

# Find contours based on Edges
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30] #sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
NumberPlateCnt = None #we currently have no Number plate contour

# loop over our contours to find the best possible approximate contour of number plate
count = 0
for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  # Select the contour with 4 corners
            NumberPlateCnt = approx #This is our approx Number Plate Contour
            x,y,w,h = cv2.boundingRect(c)
            ROI = img[y:y+h, x:x+w]
            break

if NumberPlateCnt is not None:
    # Drawing the selected contour on the original image
    cv2.drawContours(image, [NumberPlateCnt], -1, (0,255,0), 3)
print(NumberPlateCnt)

fig, ax = plt.subplots(2, 2, figsize=(10,7))
ax[0,0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax[0,0].set_title("Detected license plate")
# Find bounding box and extract ROI (Region of Interest)
ax[0,1].imshow(ROI)
ax[0,1].set_title("Extracted license plate")

'''--------------------------------------------------------------------------------------------------------------'''

mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [NumberPlateCnt], 0, 255, -1)

#cv2.bitwise_and: Applies the mask to the original image, keeping only the region of interest (the license plate or detected object).
new_image = cv2.bitwise_and(image, image, mask=mask)
ax[1,0].imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
ax[1,0].set_title("bitwise")

#(x, y) = np.where(mask == 255)
#(x1, y1) = (np.min(x), np.min(y))
#(x2, y2) = (np.max(x), np.max(y))
#cropped_image = gray[x1:x2 + 3, y1:y2 + 3]
#plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
#plt.title("cropped_image")
#plt.show()

reader = easyocr.Reader(['en'])
#result = reader.readtext(cropped_image)
result = reader.readtext(ROI)
#print("result: ",result)
#print("result[0][0]: ",result[0][0])
print("result[0][1]: ",result[0][1])
text = result[0][1]
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(image, text=text, org=(approx[1][0][1], approx[0][0][0] + 70), fontFace=font, fontScale=1,color=(0, 255, 0), thickness=2)
#res = cv2.rectangle(image, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)
ax[1,1].imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
ax[1,1].set_title("Detection: "+text)
fig.tight_layout()
plt.show()
