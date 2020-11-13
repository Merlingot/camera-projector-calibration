#!/usr/local/bin/python3
# RUN FROM /unwrapping DIRECTORY ONLY

import numpy as np
from cv2 import *
import os
import sys
import matplotlib.pyplot as plt
import glob
import pymba
from pymba import Vimba, VimbaException, Frame
# Personal modules
import util
from camera import frame_to_image


#### PARAMETRES #########################################
SERIE="serie_3"
# Numéro de la caméra
CAM_OSNUMBER=1
# Dimension du projecteur
projector_width = int(1920); projector_heigth = int(1200);
# Dimensions du first monitor (MacBook)
monitor_width = int(1440); monitor_heigth = int(900);
# Faire les directories:
wd="../camera-projector-calibration/"
#########################################################

imgName = "img_{:04d}.png"
fringeName= "phaseshift_{:03d}.png"
datafolder='data/{}/'.format(SERIE)

outputNoFringePath=os.path.join(wd,"{}nofringe/".format(datafolder))
outputCapturePath=os.path.join(wd,"{}scan_3channels/".format(datafolder))
fringesPath=os.path.join(wd,"unwrapping/fringes/")
output_paths = [outputNoFringePath,outputCapturePath]
util.outputClean(output_paths)

# -----------------------------------------------------------------------------
# Lire les patterns :
nb = len(glob.glob(fringesPath + "*.png"))
patterns =[]
for i in range(nb):
    patterns.append(imread( fringesPath + fringeName.format(i)))

# Test patterns
test_pattern=patterns[0]
white_pattern=np.ones(test_pattern.shape)*255

# Mesures
with Vimba() as vimba:
    # A. Ouvrir la camera
    cam=vimba.camera(CAM_OSNUMBER) #numero de camera
    cam.open()
    cam.arm('SingleFrame')

    # B. Ouvrir l'ecran:
    namedWindow("pattern", WINDOW_NORMAL);
    moveWindow("pattern", 0, 0);
    moveWindow("pattern", 0, -monitor_heigth);
    imshow("pattern", test_pattern);
    waitKey(0)
    setWindowProperty("pattern", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
    print('->Enter')
    waitKey(0)
    waitKey(1000)

    #C. Prendre une photo sans frange
    imshow("pattern", white_pattern);
    waitKey(500)
    frame = cam.acquire_frame()
    image = frame_to_image(frame)
    imwrite( outputNoFringePath+"noFringe" + ".png", cvtColor(image, COLOR_BGR2GRAY));
    waitKey(1000)
    #D. Prendre les photos de franges
    for i in range(len(patterns)):
        imshow("pattern", patterns[i]);
        waitKey(300)
        frame = cam.acquire_frame()
        image = frame_to_image(frame)
        waitKey(50)
        imwrite( outputCapturePath+imgName.format(i), cvtColor(image, COLOR_BGR2GRAY)  )

# Fermer la caméra et le display window
camera.disarm()
camera.close()
imshow("pattern", white_pattern);
waitKey(50)
destroyAllWindows()
