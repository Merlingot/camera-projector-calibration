# RUN FROM /unwrapping DIRECTORY ONLY
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# from skimage.io import imread, imsave
import glob
import os

scan="scan"
if not os.path.exists(scan):
    os.makedirs(scan)
else:
    for file in os.scandir(scan):
        if file.name.endswith(".png"):
            os.unlink(file.path)

nb = len(glob.glob("scan_3channels/*.png"))

for i in range(nb):

    img3= cv.imread("scan_3channels/img_{:04d}.png".format(i))

    print("img_{:04d}.png".format(i) + "-->" + "img_{:04}.png".format(i))

    img = np.mean(img3, axis=2).astype(np.uint8)

    cv.imwrite("scan/img_{:04d}.png".format(i), img)
