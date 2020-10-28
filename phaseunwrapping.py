#!/usr/local/bin/python3

import numpy as np
from cv2 import *
import os
import sys


# Faire les directories:
wd=os.getcwd()
test=os.path.join(wd,"test/")
outputCapturePath=os.path.join(wd,"output/capture/")
outputPatternPath=os.path.join(wd,"output/pattern/")
outputWrappedPath=os.path.join(wd,"output/wrapped/")
outputUnwrappedPath=os.path.join(wd,"output/unwrapped/")
reliabilitiesPath=os.path.join(wd,"output/reliabilities/")
output_paths = [test,outputPatternPath, outputPatternPath , outputWrappedPath, outputUnwrappedPath, reliabilitiesPath]
for path in output_paths:
    if not os.path.exists(path):
        os.makedirs(path)


# La camera ----------------
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
params.methodId=0
params.nbrOfPeriods=4;
params.setMarkers = True
params.horizontal = True
params.methodId = structured_light.PSP
params.shiftValue = 2.0 * np.pi / 3.0
params.nbrOfPixelsBetweenMarkers = 70
# Generate sinusoidal patterns
sinus=structured_light.SinusoidalPattern_create(params)
_, patternImages = sinus.generate()

# Enregistrer les patterns -----
for i in range(0, len(patternImages)):
    name = i + 1;
    imwrite(outputPatternPath + "pattern_{}".format(name) + ".png", patternImages[i]);
# ------------------------------

# MESURES --------------------------------------------------------------------
# A. Prendre une photo du damier sans frange
cap = VideoCapture(CAM_OSNUMBER)
if( not cap.isOpened() ):
    print( "Camera could not be opened")
    pass
ret, frame = cap.read()
retval=imwrite( test+"image_sans_frange" + ".png", frame);

# B. Prendre les photos de franges
# B.1. Afficher les franges et enregistrer les captures
namedWindow("pattern", WINDOW_NORMAL);
setWindowProperty("pattern", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
moveWindow("pattern", 0, 0);
imshow("pattern", patternImages);
waitKey(0)

nbrOfImages = 6;
count = 0;
captureImages=[]
while( count < nbrOfImages ):
    for i in range( 0, len(patternImages) ):
        imshow("pattern", patternImages[i]);
        waitKey(30)
        _, frame = cap.read()
        captureImages.append(frame)
        count += 1

# Fermer la caméra et le display window
cap.release()
moveWindow("pattern", 0, monitor_width);
destroyAllWindows()



greyImages=[]
for i in len(captureImages):
    greyImages.append( cvtColor(captureImages[i], COLOR_BGR2GRAY) )

#--------------------------------------------------------------------

# UNWRAPPING  -------------------------------------------------------
camSize = (captureImages[0].shape[0],captureImages[0].shape[1] )
paramsUnwrapping = phase_unwrapping_HistogramPhaseUnwrapping_Params()
paramsUnwrapping.height=projector_heigth;
paramsUnwrapping.width=projector_width;
phaseUnwrapping= phase_unwrapping_HistogramPhaseUnwrapping.create(paramsUnwrapping)



for  i in range( 1, nbrOfImages - 2):
    print('allo')

        if params.methodId == structured_light.FTP:

            for( i in range( 0, nbrOfImages) ):
                #We need three images to compute the shadow mask, as described in the reference paper
                # even if the phase map is computed from one pattern only
                captures=[]
                if( i == nbrOfImages - 2 ):
                    captures.append(greyImages[i]);
                    captures.append(greyImages[i-1]);
                    captures.append(greyImages[i+1]);
                else if( i == nbrOfImages - 1 ):
                    captures.push_back(greyImages[i]);
                    captures.push_back(greyImages[i-1]);
                    captures.push_back(greyImages[i-2]);
                else:
                    captures.push_back(greyImages[i]);
                    captures.push_back(greyImages[i+1]);
                    captures.push_back(greyImages[i+2]);

            sinus.computePhaseMap(captureImages)
            sinus.unwrapPhaseMap(wrappedPhaseMap, unwrappedPhaseMap, camSize, shadowMask);

                    phaseUnwrappingunwrapPhaseMap(wrappedPhaseMap, unwrappedPhaseMap, shadowMask);
                    Mat reliabilities, reliabilities8;
                    phaseUnwrapping->getInverseReliabilityMap(reliabilities);
                    reliabilities.convertTo(reliabilities8, CV_8U, 255,128);
                    ostringstream tt;
                    tt << i;
                    imwrite(reliabilitiesPath + "rel_{}".format(i) + ".png", reliabilities8);

                    if( !outputUnwrappedPhasePath.empty() )
                    {
                        ostringstream name;
                        name << i;
                        imwrite(outputUnwrappedPhasePath + "_FTP_" + name.str() + ".png", unwrappedPhaseMap8);
                    }
                    if( !outputWrappedPhasePath.empty() )
                    {
                        ostringstream name;
                        name << i;
                        imwrite(outputWrappedPhasePath + "_FTP_" + name.str() + ".png", wrappedPhaseMap8);
                    }
                }
                break;

    else:

        captures=[];
        for j in range(0,3):
            captures.append(greyImages[i+j]);

        # Wrap/Unwrap ------------
        wrappedPhaseMap, shadowMask = sinus.computePhaseMap(captures)
        print("wrappedPhaseMap succes!")
        wrappedPhaseMapScaled = convertScaleAbs(wrappedPhaseMap, 1.0, 128)
        wrappedPhaseMap8 = wrappedPhaseMapScaled.astype(np.uint8)
        shadowMaskScaled = convertScaleAbs(shadowMask, 1.0, 128)
        shadowMask8 = shadowMaskScaled.astype(np.uint8)
        # ----------------------

        # Enregistrement ------------
        # Wrapped Phase Map
        if( params.methodId == structured_light.PSP ):
            imwrite(outputWrappedPath + "_PSP_{}".format(i) + ".png", wrappedPhaseMap8);
        else:
            imwrite(outputWrappedPath + "_FAPS_{}".format(i)  + ".png", wrappedPhaseMap8);
        # ShadowMask
        if( params.methodId == structured_light.PSP ):
            imwrite(outputWrappedPath + "sm_PSP_{}".format(i) + ".png", shadowMask8);
        else:
            imwrite(outputWrappedPath + "sm_FAPS_{}".format(i)  + ".png", shadowMask8);
        # Capture images
        if( params.methodId == structured_light.PSP ):
            imwrite(outputCapturePath + "_PSP_{}".format(i)+ ".png", captureImages[i]);
        else:
            imwrite(outputCapturePath + "_FAPS_{}".format(i) + ".png", captureImages[i]);
        if( i == nbrOfImages - 3 ):
            if( params.methodId == structured_light.PSP ):
                nameBis = i+1;
                nameTer = i+2;
                imwrite(outputCapturePath + "_PSP_{}".format(nameBis) + ".png", captureImages[i+1]);
                imwrite(outputCapturePath + "_PSP_{}".format(nameTer) + ".png", captureImages[i+2]);
            else:
                nameBis = i+1;
                nameTer = i+2;
                imwrite(outputCapturePath + "_FAPS_{}".format(nameBis)+ ".png", captureImages[i+1]);
                imwrite(outputCapturePath + "_FAPS_{}".format(nameTer) + ".png", captureImages[i+2]);
        print("saved")
        # ----------------------

        # Unwrap ---------------
        sinus.unwrapPhaseMap(wrappedPhaseMap, camSize, shadowMask)
        print("unwrappedPhaseMap succes!")
        unwrappedPhaseMapScaled = convertScaleAbs(unwrappedPhaseMap, 255, 128)
        unwrappedPhaseMap8 = unwrappedPhaseMapScaled.astype(np.uint8)
        # ----------------------

        # Enregistrement ------------
        # Unwrapped Phase Map
        if( params.methodId == structured_light.PSP ):
            imwrite(outputUnwrappedPath + "_PSP_{}".format(i) + ".png", unwrappedPhaseMap8);
        else:
            imwrite(outputUnwrappedPath + "_FAPS_{}".format(i)  + ".png", unwrappedPhaseMap8);
        # --------------------------


print("Done")
--------------------------------------------------------------------
