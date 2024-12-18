import os
import numpy as np
import pandas as pd
import cv2
import imutils
import matplotlib.pyplot as plt
import easyocr
import glob

def import_image_folder(folder_path):
    image_folder=glob.glob(folder_path)
    return image_folder

def find_contours(image):
    input_image=image
    image = cv2.imread(image)
    if image is None:
        print(f"[Error] Unable to read image at {image}")
        return False
    else:
        print(f"[Info] Start processing {input_image}")

    # Resize image
    image = imutils.resize(image, width=500)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the original image
    #fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    #ax[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #ax[0, 0].set_title('Original Image')

    # RGB to Gray scale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #ax[0, 1].imshow(gray, cmap='gray')
    #ax[0, 1].set_title('Grayscale Conversion')

    # Noise removal with iterative bilateral filter(removes noise while preserving edges)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    #ax[1, 0].imshow(gray, cmap='gray')
    #ax[1, 0].set_title('Bilateral Filter')

    # Find Edges of the grayscale image
    edged = cv2.Canny(gray, 170, 200)
    #ax[1, 1].imshow(edged, cmap='gray')
    #ax[1, 1].set_title('Canny Edges')

    #fig.tight_layout()
    #plt.show()

    # Find contours based on Edges
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]  # sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
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
    #print(f"[DEBUG] {input_image} NumberPlateCnt:\n{NumberPlateCnt}")
    try:
        if NumberPlateCnt is None:
            print(f"[Error] Contour is {NumberPlateCnt} : {input_image}")
            return False
        elif NumberPlateCnt is not None:
            # Drawing the selected contour on the original image
            cv2.drawContours(image, [NumberPlateCnt], -1, (0, 255, 0), 3)
            #show_detection(gray,approx,NumberPlateCnt,ROI,image)
            return show_detection(gray,approx,NumberPlateCnt,ROI,image,input_image)
    except Exception as e:
        print(f"[ERROR] No contours found, {e}")
        return False

def show_detection(gray,approx,NumberPlateCnt,ROI,image,input_image):
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
    result = reader.readtext(ROI)

    try:
        print(f"Detection: {result[0][1]}")
    except Exception as e:
        print(f"[Error] {e}: {input_image}")
        return False
    text = result[0][1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    res = cv2.putText(image, text=text, org=(approx[1][0][1], approx[0][0][0] + 70), fontFace=font, fontScale=1,
                      color=(0, 255, 0), thickness=2)
    # res = cv2.rectangle(image, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)
    ax[1, 1].imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    ax[1, 1].set_title("Detection: " + text)
    fig.tight_layout()
    plt.show()



def main(folder_path,image_format):
    image_path = import_image_folder(folder_path+image_format)
    successful_detection=[]
    failed_detection=[]
    for image in image_path:
        success = find_contours(image)
        if success == False:
            failed_detection.append(image)
            #print(f"[Error] Skipping {image_path} due to processing failure.")

        else:
            successful_detection.append(image)
    print("========== Result ==========")
    print(f"Successful: {successful_detection}")
    print(f"Failed: {failed_detection}")



#===== Configs =====
#folder_path="material/successful_detection"
folder_path="material/failed_detection"
image_format="/*"

#===== main =====
if __name__ == "__main__":
    main(folder_path,image_format)




