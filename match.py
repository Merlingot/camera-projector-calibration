#!/usr/local/bin/python3

import numpy as np
from cv2 import *
import os
import sys



# PARAMETRES (MEMES QUE POUR LA CAPTURE!)

def main(folder, outputCapturePath, outputPatternPath, projector_width,projector_heigth,params, nb_patterns, periods, damiers, directions,
directions_s):

    # Faire les directories:
    wd=os.getcwd()
    outputWrappedPath=os.path.join(wd,"{}/wrapped/".format(folder))
    outputUnwrappedPath=os.path.join(wd,"{}/unwrapped/".format(folder))
    reliabilitiesPath=os.path.join(wd,"{}/reliabilities/".format(folder))
    outputMatchPath=os.path.join(wd,"{}/match/".format(folder))
    output_paths = [outputWrappedPath, outputUnwrappedPath, reliabilitiesPath, outputMatchPath]
    for path in output_paths:
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            for file in os.scandir(path):
                if file.name.endswith(".png"):
                    os.unlink(file.path)
    #--------------------------------------------------------------------

    # UNWRAPPING  -------------------------------------------------------
    nbrOfImages=3

    for damier in damiers:
        patternImages=[]
        captures=[]
        for direction,d in zip(directions,directions_s):
            params.horizontal = direction
            for period in periods:
                params.nbrOfPeriods=period
                sinus=structured_light.SinusoidalPattern_create(params)
                for i in range(0, nb_patterns):
                    name = i + 1;
                    patternImages.append( imread(outputPatternPath + "pattern_{}_{}_no{}".format(d, period,name) + ".png", IMREAD_GRAYSCALE));
                    captures.append( imread(outputCapturePath+"damier_{}_pattern_{}.png".format(damier, i), IMREAD_GRAYSCALE))
        # Wrap ------------
        wrappedPhaseMap, shadowMask = sinus.computePhaseMap(captures)
        # Enregistrement ------------
        wrappedPhaseMapScaled = convertScaleAbs(wrappedPhaseMap, 1.0, 128)
        wrappedPhaseMap8 = wrappedPhaseMapScaled.astype(np.uint8)
        shadowMaskScaled = convertScaleAbs(shadowMask, 1.0, 128)
        shadowMask8 = shadowMaskScaled.astype(np.uint8)
        # ----------------------

        # Unwrap ---------------
        camSize = (captures[0].shape[0],captures[0].shape[1])
        unwrappedPhaseMap = sinus.unwrapPhaseMap(wrappedPhaseMap, camSize, shadowMask)
        print("unwrappedPhaseMap succes!")
        unwrappedPhaseMapScaled = convertScaleAbs(unwrappedPhaseMap, 255, 128)
        unwrappedPhaseMap8 = unwrappedPhaseMapScaled.astype(np.uint8)

        # Match -----------------
        projSize=(patternImages[0].shape[0],patternImages[0].shape[1])
        projwrappedPhaseMap, projshadowMask = sinus.computePhaseMap(patternImages)
        projunwrappedPhaseMap = sinus.unwrapPhaseMap(projwrappedPhaseMap, projSize)
        matches	= sinus.findProCamMatches(patternImages, unwrappedPhaseMap)

        # Wrapped
        imwrite(outputWrappedPath + "wrapped_damier{}_{}_{}".format(damier,d,period)  + ".png", wrappedPhaseMap8);
        # ShadowMask
        imwrite(outputWrappedPath + "shadowMask_damier{}_{}_{}".format(damier,d,period) + ".png", shadowMask8);
        print("saved")
        # Unwrapped
        imwrite(outputUnwrappedPath + "unwrapped_damier{}_{}_{}".format(damier, d, period) + ".png", unwrappedPhaseMap8);
        # Cam matches
        imwrite(outputMatchPath + "procamMatch{}_{}_{}".format(damier, d, period) + ".png", matches);
        # ----------------------
