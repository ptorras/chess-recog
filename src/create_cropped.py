import cv2
from matplotlib import pyplot as plt
import numpy as np

# open the imge
def crop_img(img):
    blur = cv2.pyrMeanShiftFiltering(img, 11, 21)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # sort by area from largest to smallest
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # remove all the contours that are not rectangles
    cnts = [cnt for cnt in cnts if len(cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)) == 4]

    # remove any contour that is a horizontal rectangle instead of a vertical rectangle
    cnts = [cnt for cnt in cnts if cv2.boundingRect(cnt)[2] < 1.5 * cv2.boundingRect(cnt)[3]][0]

    # crop the image using the upper and lower bounds of the rectangle
    x, y, w, h = cv2.boundingRect(cnts)
    result = img[y:y+h, :]

    # save the image
    return result


import os 

for root, dir, files in os.walk(r'C:\Users\Maria\Desktop\HackathonCVC\Games'):
    for file in files:
        if file.endswith('.jpg'):
            print(os.path.join(root, file))
            img = cv2.imread(os.path.join(root, file))
            result = crop_img(img)
            print(os.path.join(root, file[:-4]+ '_cropped.jpg'))
            cv2.imwrite(os.path.join(root, file[:-4]+ '_cropped.jpg'), result)
            break