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
from keras import layers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Input, Dropout
from tensorflow.python.keras.models import Model, Sequential
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
tf.config.list_physical_devices("GPU")


'''Resolve the issue of 
AttributeError: module 'tensorflow.python.distribute.input_lib' has no attribute 'DistributedDatasetInterface
from 
File "/opt/anaconda3/lib/python3.12/site-packages/tensorflow/python/keras/engine/data_adapter.py", line 1696, in _is_distributed_dataset
    return isinstance(ds, input_lib.DistributedDatasetInterface)'''
from tensorflow.python.keras.engine import data_adapter
def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)
data_adapter._is_distributed_dataset = _is_distributed_dataset


def import_image(input_image):
    input_file = input_image
    if os.path.isfile(input_file):
        print(f"The file '{input_file}' exists.")
        image = cv2.imread(input_file)
        return image
    else:
        print("[ERROR]", f"The file '{input_file}' does not exist.")
        exit()

def find_contours_by_joe(image):
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
    #plt.show()

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
    return gray,approx,NumberPlateCnt,ROI,image

def show_detection(gray,approx,NumberPlateCnt,ROI,image):
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    ax[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[0, 0].set_title("Detected license plate")
    # Find bounding box and extract ROI (Region of Interest)
    ax[0, 1].imshow(ROI)
    ax[0, 1].set_title("Extracted license plate")

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [NumberPlateCnt], 0, 255, -1)

    # cv2.bitwise_and: Applies the mask to the original image, keeping only the region of interest (the license plate or detected object).
    new_image = cv2.bitwise_and(image, image, mask=mask)
    ax[1, 0].imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    ax[1, 0].set_title("bitwise")

    reader = easyocr.Reader(['en'])
    # result = reader.readtext(cropped_image)
    result = reader.readtext(ROI)
    # print("result: ",result)
    # print("result[0][0]: ",result[0][0])
    print("result[0][1]: ", result[0][1])
    text = result[0][1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    res = cv2.putText(image, text=text, org=(approx[1][0][1], approx[0][0][0] + 70), fontFace=font, fontScale=1,color=(0, 255, 0), thickness=2)
    # res = cv2.rectangle(image, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)
    ax[1, 1].imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    ax[1, 1].set_title("Detection: " + text)
    fig.tight_layout()
    #plt.show()

# Distance between (x1, y1) and (x2, y2)
def dist(x1, x2, y1, y2):
    return ((x1-x2)**2+(y1-y2)**2)**0.5


# Match contours to license plate or character template
def find_contours(dimensions, img):
    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]

    # Check largest 5 or  15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

    ii = cv2.imread('contour.jpg')

    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs:
        # detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)

        # checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height:
            x_cntr_list.append(
                intX)  # stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((44, 24))
            # extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY + intHeight, intX:intX + intWidth]
            char = cv2.resize(char, (20, 40))

            cv2.rectangle(ii, (intX, intY), (intWidth + intX, intY + intHeight), (50, 21, 200), 2)
            plt.imshow(ii, cmap='gray')
            plt.title('Predict Segments')

            # Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy)  # List that stores the character's binary image (unsorted)

    # Return characters on ascending order with respect to the x-coordinate (most-left character first)

    #plt.show()
    # arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])  # stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res

# Find characters in the resulting images
def segment_characters(image) :

    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/6,
                       LP_WIDTH/2,
                       LP_HEIGHT/10,
                       2*LP_HEIGHT/3]
    plt.imshow(img_binary_lp, cmap='gray')
    plt.title('Contour')
    #plt.show()
    cv2.imwrite('contour.jpg',img_binary_lp)

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_binary_lp)

    return char_list


input_image="/Users/Joe/Documents/Uni_Taipei/Image_Process/License_Plate_Detection/POC/material/successful_detection/plate01.jpg"
#image=import_image(input_image)
#gray,approx,NumberPlateCnt,ROI,image=find_contours_by_joe(image)
#show_detection(gray,approx,NumberPlateCnt,ROI,image)


input_file=input_image
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
#plt.show()

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

#-------------------- new --------------------
idx=0
m=0
# To find the index of coordinate with maximum y-coordinate
for i in range(4):
    if NumberPlateCnt[i][0][1]>m:
        idx=i
        m=NumberPlateCnt[i][0][1]

# Assign index to the previous coordinate
if idx==0:
    pin=3
else:
    pin=idx-1

# Assign index to the next coordinate
if idx==3:
    nin=0
else:
    nin=idx+1

# Find distances between the acquired coordinate and its previous and next coordinate
p=dist(NumberPlateCnt[idx][0][0], NumberPlateCnt[pin][0][0], NumberPlateCnt[idx][0][1], NumberPlateCnt[pin][0][1])
n=dist(NumberPlateCnt[idx][0][0], NumberPlateCnt[nin][0][0], NumberPlateCnt[idx][0][1], NumberPlateCnt[nin][0][1])

# The coordinate that has more distance from the acquired coordinate is the required second bottom-most coordinate
if p>n:
    if NumberPlateCnt[pin][0][0]<NumberPlateCnt[idx][0][0]:
        left=pin
        right=idx
    else:
        left=idx
        right=pin
    d=p
else:
    if NumberPlateCnt[nin][0][0]<NumberPlateCnt[idx][0][0]:
        left=nin
        right=idx
    else:
        left=idx
        right=nin
    d=n
print("left=",left,", right=",right)

left_x=NumberPlateCnt[left][0][0]
left_y=NumberPlateCnt[left][0][1]
right_x=NumberPlateCnt[right][0][0]
right_y=NumberPlateCnt[right][0][1]
print(left_x, left_y, right_x, right_y)

# Finding the angle of rotation by calculating sin of theta
opp=right_y-left_y
hyp=((left_x-right_x)**2+(left_y-right_y)**2)**0.5
sin=opp/hyp
theta=math.asin(sin)*57.2958

# Rotate the image according to the angle of rotation obtained
image_center = tuple(np.array(ROI.shape[1::-1]) / 2)
rot_mat = cv2.getRotationMatrix2D(image_center, theta, 1.0)
result = cv2.warpAffine(ROI, rot_mat, ROI.shape[1::-1], flags=cv2.INTER_LINEAR)

# The image can be cropped after rotation( since rotated image takes much more height)
if opp>0:
    h=result.shape[0]-opp//2
else:
    h=result.shape[0]+opp//2

result=result[0:h, :]
plt.imshow(result)
plt.title("Plate obtained after rotation")
#plt.show()

char=segment_characters(result)

for i in range(len(char)):
    plt.subplot(1, len(char), i+1)
    plt.imshow(char[i], cmap='gray')
    plt.axis('off')
#plt.show()

#---------- create a Neural Network ----------
# 設定批量生成器
train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.1, height_shift_range=0.1)
path = 'data'
train_generator = train_datagen.flow_from_directory(
        path+'/train',  # this is the target directory
        target_size=(28,28),  # all images will be resized to 28x28
        batch_size=1,
        class_mode='sparse') # Default 'categorical':2D one-hot encoding / 'sparse':1D 整數編碼標籤

validation_generator = train_datagen.flow_from_directory(
        path+'/val',  # this is the target directory
        target_size=(28,28),  # all images will be resized to 28x28
        batch_size=1,
        class_mode='sparse')

K.clear_session()
model = Sequential()
model.add(Conv2D(16, (22,22), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (16,16), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (8,8), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (4,4), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(36, activation='softmax'))
#model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0001), metrics='accuracy')
#model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics='accuracy')
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics='accuracy')
model.summary()

# 嘗試從 train_generator 中獲取一批數據
print("[DEBUG]")
try:
    # 使用 next() 取出一批數據
    batch_x, batch_y = next(train_generator)

    # 顯示數據的形狀
    print("Batch X (input data) shape:", batch_x.shape)
    print("Batch Y (target labels) shape:", batch_y.shape)

    # 檢查數據是否包含 NaN 或空值
    print("NaN in X:", np.isnan(batch_x).any())
    print("NaN in Y:", np.isnan(batch_y).any())

    # 顯示部分數據內容（可選）
    print("Sample input data (X):", batch_x[0])
    print("Sample target labels (Y):", batch_y[0])

except StopIteration:
    print("Error: The train_generator has no more data to yield.")
except Exception as e:
    print("An error occurred while fetching data from train_generator:", e)

batch_size = 25
result = model.fit(train_generator,
      steps_per_epoch = train_generator.samples // batch_size,
      validation_data = validation_generator,
      epochs = 25,
      verbose=1,
      workers=1,
      use_multiprocessing=False
      )


'''#from GPT
K.clear_session()
model = Sequential()
model.add(Conv2D(16, (3,3), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(36, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
model.summary()

batch_size = 16
result = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    epochs=25,
    verbose=2,
    workers=1,
    use_multiprocessing=False
)
#'''


fig = plt.figure(figsize=(14,5))
grid=gridspec.GridSpec(ncols=2,nrows=1,figure=fig)
fig.add_subplot(grid[0])
plt.plot(result.history['accuracy'], label='training accuracy')
plt.plot(result.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

fig.add_subplot(grid[1])
plt.plot(result.history['loss'], label='training loss')
plt.plot(result.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

# Save the weights
model.save_weights('./checkpoints/my_checkpoint')
# Create a new model instance
loaded_model = Sequential()
loaded_model.add(Conv2D(16, (22,22), input_shape=(28, 28, 3), activation='relu', padding='same'))
loaded_model.add(Conv2D(32, (16,16), input_shape=(28, 28, 3), activation='relu', padding='same'))
loaded_model.add(Conv2D(64, (8,8), input_shape=(28, 28, 3), activation='relu', padding='same'))
loaded_model.add(Conv2D(64, (4,4), input_shape=(28, 28, 3), activation='relu', padding='same'))
loaded_model.add(MaxPooling2D(pool_size=(4, 4)))
loaded_model.add(Dropout(0.4))
loaded_model.add(Flatten())
loaded_model.add(Dense(128, activation='relu'))
loaded_model.add(Dense(36, activation='softmax'))

# Restore the weights
loaded_model.load_weights('checkpoints/my_checkpoint')



#--------------------------------------------------------------------------------------------------------------
'''
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
'''