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

# Faire les directories:
folder='output'
wd=os.getcwd()
noFringePath=os.path.join(wd,"{}/nofringe/".format(folder))
outputCapturePath=os.path.join(wd,"{}/capture/".format(folder))
outputPatternPath=os.path.join(wd,"{}/pattern/".format(folder))
output_paths = [noFringePath,outputCapturePath, outputPatternPath]
for path in output_paths:
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        for file in os.scandir(path):
            if file.name.endswith(".png"):
                os.unlink(file.path)


# Camera et moniteurs ----------------
CAM_OSNUMBER=1
# Dimensions du first monitor
monitor_width = int(2880); monitor_heigth = int(1800);
# Dimensions du first monitor
projector_width = int(1280) ; projector_heigth = int(720);


# Sinusoidal Patterns------------------------------------
# Set patterns parameters
params = structured_light_SinusoidalPattern_Params()
params.width=projector_width
params.height=projector_heigth
params.nbrOfPeriods=int(3);
params.setMarkers = True
params.horizontal = True
params.methodId = structured_light.PSP
params.shiftValue = 2.0 * np.pi / 3.0
params.nbrOfPixelsBetweenMarkers = 70
# Generate sinusoidal patterns
directions=[False, True]
directions_s=['v','h']
periods=[3,5,7,9,11,13]
nb_patterns=3
for direction,d in zip(directions,directions_s):
    params.horizontal = direction
    for period in periods:
        params.nbrOfPeriods=period
        sinus=structured_light.SinusoidalPattern_create(params)
        _, patternImages = sinus.generate()
        # Enregistrer les patterns ---------------------------
        for i in range(0, nb_patterns):
            name = i + 1;
            imwrite(outputPatternPath + "pattern_{}_{}_no{}".format(d, period,name) + ".png", patternImages[i]);
        # # ----------------------------------------------------

# MESURES --------------------------------------------------------------------

# Lire les images
patternImages=[] #Tous les patterns h/v et nbres de periodes
for direction,d in zip(directions,directions_s):
    for period in periods:
        for i in range(0, nb_patterns):
            name = i + 1;
            patternImages.append(imread(outputPatternPath + "pattern_{}_{}_no{}".format(d, period,name) + ".png"))

# A. Ouvrir la camera
cap = VideoCapture(CAM_OSNUMBER)
if( not cap.isOpened() ):
    print( "Camera could not be opened")
    pass
# Pour une position de damier:
quit="pasq"
damier=0
while quit!="q":
    #A. Prendre une photo sans frange
    ret, frame = cap.read()
    retval=imwrite( noFringePath+"sans_frange_{}".format{damier} + ".png", cvtColor(frame, COLOR_BGR2GRAY));
    # B. Prendre les photos de franges
    # B.1 Ouvrir une nouvelle fenetre:
    namedWindow("pattern", WINDOW_NORMAL);
    moveWindow("pattern", 0, -monitor_width);
    setWindowProperty("pattern", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
    for i in range( 0, len(patternImages) ):
        imshow("pattern", patternImages[i]);
        waitKey(10)
        _, frame = cap.read()
        imwrite(outputCapturePath+"damier_{}_pattern_{}.png".format(damier, i), cvtColor(frame, COLOR_BGR2GRAY)  )
    destroyAllWindows()
    quit=input('q to quit')

# Fermer la caméra et le display window
cap.release()




# # Enregistrement des capture images ----------------------
# if( params.methodId == structured_light.PSP ):
#     imwrite(outputCapturePath + "_PSP_{}".format(i)+ ".png", captureImages[i]);
# else:
#     imwrite(outputCapturePath + "_FAPS_{}".format(i) + ".png", captureImages[i]);
# if( i == nbrOfImages - 3 ):
#     if( params.methodId == structured_light.PSP ):
#         nameBis = i+1;
#         nameTer = i+2;
#         imwrite(outputCapturePath + "_PSP_{}".format(nameBis) + ".png", captureImages[i+1]);
#         imwrite(outputCapturePath + "_PSP_{}".format(nameTer) + ".png", captureImages[i+2]);
#     else:
#         nameBis = i+1;
#         nameTer = i+2;
#         imwrite(outputCapturePath + "_FAPS_{}".format(nameBis)+ ".png", captureImages[i+1]);
#         imwrite(outputCapturePath + "_FAPS_{}".format(nameTer) + ".png", captureImages[i+2]);
# # -----------------------------------------------------------
