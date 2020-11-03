#!/usr/local/bin/python3

import numpy as np
from cv2 import *
import os
import sys
import matplotlib.pyplot as plt


def main(folder, outputCapturePath, outputPatternPath, projector_width,projector_heigth,params, nb_patterns, periods, directions,
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
    unwrappedImages = []
    for direction,d in zip(directions,directions_s):
        params.horizontal = direction
        for period in periods:
            params.nbrOfPeriods=period

            # Un sinus ---------------------------------------------
            sinus=structured_light.SinusoidalPattern_create(params)

            patternImages=[]
            captures=[]
            for i in range(0, nb_patterns):
                captures.append( imread(outputCapturePath+"capture_{}_{}_{}.png".format(d,period,i), IMREAD_GRAYSCALE))
                patternImages.append( imread(outputPatternPath+"pattern_{}_{}_{}.png".format(d,period,i), IMREAD_GRAYSCALE))

            # Bonnes Sizes:
            camSize = (captures[0].shape[1],captures[0].shape[0])
            projSize = (params.width,params.height)

            # Projecteur
            projwrappedPhaseMap, projshadowMask = sinus.computePhaseMap(patternImages)
            projunwrappedPhaseMap = projwrappedPhaseMap.copy()
            sinus.unwrapPhaseMap(projwrappedPhaseMap, projSize, projunwrappedPhaseMap, projshadowMask)
            # Camera
            wrappedPhaseMap, shadowMask = sinus.computePhaseMap(captures)
            unwrappedPhaseMap = wrappedPhaseMap.copy()
            unwrappedPhaseMap=sinus.unwrapPhaseMap(wrappedPhaseMap, camSize, unwrappedPhaseMap, shadowMask)
            plt.imshow(unwrappedPhaseMap)
            # Proj-Cam Match
            matches = sinus.findProCamMatches(projunwrappedPhaseMap, unwrappedPhaseMap)




            # Enregistrement ------------
            wrappedPhaseMapScaled = convertScaleAbs(wrappedPhaseMap, 1.0, 128)
            wrappedPhaseMap8 = wrappedPhaseMapScaled.astype(np.uint8)
            shadowMaskScaled = convertScaleAbs(shadowMask, 1.0, 128)
            shadowMask8 = shadowMaskScaled.astype(np.uint8)
            unwrappedPhaseMapScaled = convertScaleAbs(unwrappedPhaseMap, 255, 128)
            unwrappedPhaseMap8 = unwrappedPhaseMapScaled.astype(np.uint8)
            unwrappedImages.append(unwrappedPhaseMap8)

            # Wrapped
            imwrite(outputWrappedPath + "wrapped_{}_{}".format(d,period)  + ".png", wrappedPhaseMap8);
            # ShadowMask
            imwrite(outputWrappedPath + "shadowMask_{}_{}".format(d,period) + ".png", shadowMask8);
            # Unwrapped
            imwrite(outputUnwrappedPath + "unwrapped_{}_{}".format(d, period) + ".png", unwrappedPhaseMap8);
            # Cam matches
            for m in range(len(matches)):
                imwrite(outputMatchPath + "procamMatch_{}_{}".format(d, period,m) + ".png", matches[m]);
            # ----------------------
    return unwrappedImages
