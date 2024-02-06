def is_close(p1, p2, threshold=5):
    if p1 < p2 + threshold and p1 > p2 - threshold:
        return True
    else:
        return False

# x, y, w, h
def sortBB(BB):
    BB.sort(key=lambda x: x[1])

    BB_copy = BB.copy()
    for i, (bb1, bb2) in enumerate(zip(BB, BB[1:])):
        x1, y1, w1, h1 = bb1
        x2, y2, w2, h2 = bb2
        if is_close(y1, y2, 5):
            if x1 > x2: 
                BB_copy[i] = (x2, y2, w2, h2)
                BB_copy[i+1] = (x1, y1, w1, h1)
                # if bb_intersection_over_union(BB_copy[i], BB_copy[i+1]) > 0.3:
                #     BB_copy.pop(i)
    BB = BB_copy
    return BB

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
import cv2
from matplotlib import pyplot as plt
import numpy as np

# open the imge

def extractTables(img, proccessed):
    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # remove all the contours that are not rectangles
    cnts = [cnt for cnt in cnts if len(cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)) == 4]

    # remove any contour that is a horizontal rectangle instead of a vertical rectangle
    cnts = [cnt for cnt in cnts if cv2.boundingRect(cnt)[2] < 1.5 * cv2.boundingRect(cnt)[3]]

    # remove small contours
    cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > 0.1 * img.shape[0] * img.shape[1]]

    # crop the image using the rectangles
    bb = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        bb.append((x, y, w, h))

    bb.sort(key=lambda x: x[0])

    imgs = []
    proccessed_imgs = []
    for b in bb:
        x, y, w, h = b
        imgs.append(img[y:y+h, x:x+w])
        proccessed_imgs.append(proccessed[y:y+h, x:x+w])

    return imgs, proccessed_imgs
def extractSquares(img_grid, final_img):
    gray = cv2.cvtColor(img_grid, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    cnts = [cnt for cnt in cnts if cv2.boundingRect(cnt)[2] > 1 * cv2.boundingRect(cnt)[3]]

    cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > 0.005 * img.shape[0] * img.shape[1]]

    widths = []
    heights = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        widths.append(w)
        heights.append(h) 
        
    median_w, median_h = int(np.median(widths)), int(np.median(heights))

    BB = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if h*w > 1.1*median_w*median_h:
            # look if the rectangle is more or less the double of the median
            ratio = int(np.round(w/median_w))
            if ratio > 1.5:
                for i in range(ratio):
                    # cv2.rectangle(img_grid, (x + i*median_w, y), (x + (i+1)*median_w, y + h), (36, 255, 12), 2)
                    BB.append((x + i*median_w, y, median_w, h))
        
            ratio2 = int(np.round(h/median_h))
            # print(ratio2)
            if ratio2 > 1.5:
                for i in range(ratio2):
                    # cv2.rectangle(img_grid, (x, y + i*median_h), (x + w, y + (i+1)*median_h), (36, 255, 12), 2)
                    BB.append((x, y + i*median_h, w, median_h))
                    
        
        else:
            if h < 0.8*median_h:
                h = median_h
            if w < 0.8*median_w:
                w = median_w

            # cv2.rectangle(img_grid, (x, y), (x + w, y + h), (36, 255, 12), 2)
            BB.append((x, y, w, h))
        
        BB = sortBB(BB)
        
        imgs = []
        for bb in BB:
            x, y, w, h = bb
            imgs.append(final_img[y:y+h, x:x+w])
        
    return imgs

import os 
def findAllSquares(img):
    all_squares = []
    
    gray_result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(gray_result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 4)
    blur = cv2.GaussianBlur(thr, (7, 7), 0)
    ret3, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    imgs, processed = extractTables(img, thr)
    for img, proc in zip(imgs, processed):
        all_squares.append(extractSquares(img, proc))
        plt.figure(figsize=(20, 20))
        plt.imshow(all_squares[-1][-1])
        plt.show()
    return all_squares