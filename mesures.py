#!/usr/local/bin/python3
# 1) Generer des patterns
#    - horizontaux/verticaux
#    - différents numéros de périodes
# 2) enregistrer
# 3) projeter et capturer avec la camera

import numpy as np
from cv2 import *
import os
import sys
import matplotlib.pyplot as plt
# from match import main
import glob

# Faire les directories:
folder='tsaigrid_test4'
wd=os.getcwd()
noFringePath=os.path.join(wd,"{}/nofringe/".format(folder))
outputCapturePath=os.path.join(wd,"{}/capture/".format(folder))
outputPatternPath=os.path.join(wd,"{}/pattern/".format(folder))
output_paths = [noFringePath,outputCapturePath, outputPatternPath]
for path in output_paths:
    if not os.path.exists(path):
        os.makedirs(path)
    # else:
    #     for file in os.scandir(path):
    #         if file.name.endswith(".png"):
    #             os.unlink(file.path)

# Camera et moniteurs ----------------
CAM_OSNUMBER=1
# Dimensions du first monitor
monitor_width = int(1440); monitor_heigth = int(900);
# Dimensions du first monitor
projector_width = int(800) ; projector_heigth = int(600);

# Sinusoidal Patterns------------------------------------
# Set patterns parameters
params = structured_light_SinusoidalPattern_Params()
params.width=projector_width
params.height=projector_heigth
params.nbrOfPeriods=0;
params.setMarkers = True
params.horizontal = True
params.methodId = structured_light.FAPS
params.shiftValue = 2.0 * np.pi / 3.0
params.nbrOfPixelsBetweenMarkers = 70
# Generate sinusoidal patterns
directions=[False, True]
directions_s=['v','h']
periods=[4,6,8,10,12]
nb_patterns=3

# -----------------------------------------------------------------------------
# Generer les patterns
for direction,d in zip(directions,directions_s):
    params.horizontal = direction
    for period in periods:
        params.nbrOfPeriods=period
        sinus=structured_light.SinusoidalPattern_create(params)
        _, patterns = sinus.generate()
        # Enregistrer les patterns ---------------------------
        for i in range(0, nb_patterns):
            imwrite(outputPatternPath + "pattern_{}_{}_{}.png".format(d,period,i), patterns[i]);

test_pattern=patterns[0]

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
white_pattern=np.ones(test_pattern.shape)*255
imshow("pattern", white_pattern);
waitKey(500)
ret, frame = cap.read()
retval=imwrite( noFringePath+"sansfrange" + ".png", cvtColor(frame, COLOR_BGR2GRAY));
waitKey(1000)
#2.Prendre les photos de franges
for direction,d in zip(directions,directions_s):
    params.horizontal = direction
    for period in periods:
        for i in range(0, nb_patterns):
            pattern = imread(outputPatternPath + "pattern_{}_{}_{}.png".format(d,period,i))
            imshow("pattern", pattern);
            waitKey(100)
            _, frame = cap.read()
            imwrite( outputCapturePath+"capture_{}_{}_{}.png".format(d,period,i), cvtColor(frame, COLOR_BGR2GRAY)  )

# Fermer la caméra et le display window
cap.release()
imshow("pattern", white_pattern);
waitKey(1000)
destroyAllWindows()
# -----------------------------------------------------------------------------


# # UNWRAPPING  -------------------------------------------------------

imgs = main(folder, outputCapturePath, outputPatternPath, projector_width,projector_heigth,params, nb_patterns, periods, directions,
directions_s)
