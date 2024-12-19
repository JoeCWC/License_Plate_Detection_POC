import os
import numpy as np
import pandas as pd
import cv2
import imutils
import matplotlib.pyplot as plt
import easyocr
import glob
import logging

def import_image_folder(folder_path):
    try:
        image_folder=glob.glob(folder_path)
        return image_folder
    except Exception as e:
        logging.error(f"{e}")

def find_contours(image,output_path):
    input_image=image
    image = cv2.imread(image)
    if image is None:
        logging.error(f"Unable to read image at {image}")
        return False
    #else:
        #logging.info(f"Start processing {input_image}")

    '''Resize image'''
    image = imutils.resize(image, width=500)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    '''Display the original image'''
    #fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    #ax[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #ax[0, 0].set_title('Original Image')

    '''RGB to Gray scale conversion'''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #ax[0, 1].imshow(gray, cmap='gray')
    #ax[0, 1].set_title('Grayscale Conversion')

    '''Noise removal with iterative bilateral filter(removes noise while preserving edges)'''
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    #ax[1, 0].imshow(gray, cmap='gray')
    #ax[1, 0].set_title('Bilateral Filter')

    '''Find Edges of the grayscale image'''
    edged = cv2.Canny(gray, 170, 200)
    #ax[1, 1].imshow(edged, cmap='gray')
    #ax[1, 1].set_title('Canny Edges')

    #fig.tight_layout()
    #plt.show()

    '''Find contours based on Edges'''
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]  # sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
    NumberPlateCnt = None  # we currently have no Number plate contour
    '''loop over our contours to find the best possible approximate contour of number plate'''
    count = 0
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  # Select the contour with 4 corners
            NumberPlateCnt = approx  # This is our approx Number Plate Contour
            x, y, w, h = cv2.boundingRect(c)
            if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
                logging.error(f"Invalid ROI: x={x}, y={y}, w={w}, h={h}")
                continue
            ROI = img[y:y + h, x:x + w]
            break
    #print(f"[DEBUG] {input_image} NumberPlateCnt:\n{NumberPlateCnt}")
    try:
        if NumberPlateCnt is None:
            logging.error(f"Contour is {NumberPlateCnt}: {input_image}")
            return False
        elif NumberPlateCnt is not None:
            '''Drawing the selected contour on the original image'''
            cv2.drawContours(image, [NumberPlateCnt], -1, (0, 255, 0), 3)
            return show_detection(gray,NumberPlateCnt,ROI,image,input_image,output_path)
    except Exception as e:
        logging.error(f"No contours found, {e}")
        return False

def show_detection(gray,NumberPlateCnt,ROI,image,input_image,output_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    #ax[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #ax[0, 0].set_title("Detected license plate")
    '''Find bounding box and extract ROI (Region of Interest)'''
    #ax[0, 1].imshow(ROI)
    #ax[0, 1].set_title("Extracted license plate")

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [NumberPlateCnt], 0, 255, -1)

    '''cv2.bitwise_and: Applies the mask to the original image, keeping only the region of interest (the license plate or detected object)'''
    new_image = cv2.bitwise_and(image, image, mask=mask)
    #ax[1, 0].imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    #ax[1, 0].set_title("bitwise")

    reader = easyocr.Reader(['en'])
    result = reader.readtext(ROI)
    #print("[DEBUG] "+result)
    x, y, w, h = cv2.boundingRect(NumberPlateCnt) # 計算多邊形的邊界框
    text_x = x  # 文字與多邊形左邊界對齊
    text_y = y + h + 25  # 文字位置在多邊形底部下方 20 像素

    try:
        text = result[0][1]
        #print(f"[DEBUG] Detection: {result[0][1]} from {input_image}")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, text=text, org=(text_x,text_y), fontFace=font,fontScale=1, color=(0, 255, 0), thickness=2)
        output = output_path+"/"+text+".jpg"
        cv2.imwrite(output, image)

        return result[0][1]
    except Exception as e:
        logging.error(f"{e}:{input_image}")
        return False
    #==========
    #ax[0, 0].imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    #ax[0, 0].set_title("Detection: " + text)
    #fig.tight_layout()
    #plt.show()

def main(folder_path,image_format,output_path):
    image_path = import_image_folder(folder_path+image_format)
    successful_detection=[]
    failed_detection=[]
    number_of_successful=0
    number_of_failed=0
    for image in image_path:
        detected_text = find_contours(image,output_path)
        if detected_text == False:
            failed_detection.append(image)
            logging.error(f"Skipping {image_path} due to processing failure.")
            number_of_failed+=1
        else:
            successful_detection.append(image)
            number_of_successful+=1
            print(f"successful: {image} : {detected_text}")
            #print(f"Number of successful: {number_of_successful}")
    print("========== Result ==========")
    print(f"Number of successful: {number_of_successful}")
    print(f"Number of failed: {number_of_failed}")

'''===== Configs ====='''
folder_path="material/successful_detection"
image_format="/*.jpg"
output_path="output"

'''===== main ====='''
if __name__ == "__main__":
    main(folder_path,image_format,output_path)




