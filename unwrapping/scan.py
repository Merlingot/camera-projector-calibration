#!/usr/local/bin/python3

import numpy as np
from cv2 import *
import os
import sys
import matplotlib.pyplot as plt
import glob

#### PARAMETRES #########################################
SERIE_NO=2
wd="/Users/mariannelado-roy/camera-projector-calibration/"
#########################################################

# Faire les directories:
imgName = "img_{:04d}.png"
fringeName= "phaseshift_{:03d}.png"
datafolder='data/serie_{}/'.format(SERIE_NO)

outputNoFringePath=os.path.join(wd,"{}nofringe/".format(datafolder))
outputCapturePath=os.path.join(wd,"{}scan_3channels/".format(datafolder))
fringesPath=os.path.join(wd,"unwrapping/fringes/")
output_paths = [outputNoFringePath,outputCapturePath]
for path in output_paths:
    if not os.path.exists(path):
        os.makedirs(path)

# Camera et moniteurs ----------------
CAM_OSNUMBER=1
# Dimensions du first monitor
monitor_width = int(1440); monitor_heigth = int(900);
# Dimensions du first monitor
projector_width = int(800) ; projector_heigth = int(600);


# -----------------------------------------------------------------------------
# Lire les patterns :
nb = len(glob.glob(fringesPath + "*.png"))
patterns =[]
for i in range(nb):
    patterns.append(imread( fringesPath + fringeName.format(i)))

# Display patterns
test_pattern=patterns[0]
white_pattern=np.ones(test_pattern.shape)*255

# A. Ouvrir la camera
cap = VideoCapture(CAM_OSNUMBER)
if( not cap.isOpened() ):
    print( "Camera could not be opened")
    pass

#B. Ouvrir l'ecran:
namedWindow("pattern", WINDOW_NORMAL);
moveWindow("pattern", 0, 0);
moveWindow("pattern", 0, -monitor_heigth);
imshow("pattern", test_pattern);
waitKey(0)
setWindowProperty("pattern", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
print('Enlever la souriiiiiiis! ->Enter')
waitKey(0)
waitKey(1000)

#1.Prendre une photo sans frange
imshow("pattern", white_pattern);
waitKey(500)
ret, frame = cap.read()
retval=imwrite( outputNoFringePath+"noFringe" + ".png", cvtColor(frame, COLOR_BGR2GRAY));
waitKey(1000)
#2.Prendre les photos de franges
for i in range(len(patterns)):
    imshow("pattern", patterns[i]);
    waitKey(300)
    _, frame = cap.read()
    waitKey(50)
    imwrite( outputCapturePath+imgName.format(i), cvtColor(frame, COLOR_BGR2GRAY)  )

# Fermer la caméra et le display window
cap.release()
imshow("pattern", white_pattern);
waitKey(50)
destroyAllWindows()
