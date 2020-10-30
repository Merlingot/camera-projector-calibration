#!/usr/local/bin/python3

seriesNO=2

import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage import maximum_filter, minimum_filter


def midpoint(img):
    maxf = maximum_filter(img, size=5)
    minf = minimum_filter(img, size=5)
    midpoint = (maxf + minf) / 2
    midpoint[midpoint < 0] = 0
    return midpoint


def confidenceMap(sgmf, name):
    img = cv2.imread(sgmf)
    red = img[:,:,1]     # CHANNEL Y
    green = img[:,:,2]   # CHANNEL X

    # Adaptive Thresholding
    green2 = cv2.adaptiveThreshold(green, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2)

    red2 = cv2.adaptiveThreshold(red, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2)

    mask=red2*green2
    mask=midpoint(mask)
    mask = mask.astype("uint8")*255
    mask =  cv2.medianBlur(mask,15)

    cv2.imwrite(name,mask)

    return 0



confidenceMap("data/serie_{}/cam_match.png".format(seriesNO), 'data/serie_{}/shadowMask.png'.format(seriesNO))
