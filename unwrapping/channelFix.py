""" Script pour prendre des images 3 channel dans un folder <dir3channel>/scan_3channels et en produire des copies noir et blanc dans un folder <dirscan>/scan """
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# from skimage.io import imread, imsave
import glob
import os

## DIRECTORY a partir de camera-projector-calibration toujours

#Folder ou prendre les franges 3 couleurs
dir3channels='./data/tests/serie_1/scan_3channels/'

#Folder ou mettre les franges noires et blanches
dirscan='./unwrapping/mini_projecteur/scan/'


if not os.path.exists(dirscan):
    os.makedirs(dirscan)
else:
    for file in os.scandir(dirscan):
        if file.name.endswith(".png"):
            os.unlink(file.path)

nb = len(glob.glob("{}*.png".format(dir3channels)))

for i in range(nb):

    img3= cv.imread("{}img_{:04d}.png".format(dir3channels,i))

    print("img_{:04d}.png".format(i) + "-->" + "img_{:04}.png".format(i))

    img = np.mean(img3, axis=2).astype(np.uint8)

    cv.imwrite("{}img_{:04d}.png".format(dirscan,i), img)
