""" calibration Zhang pour intrinsèques et avec les cibles pour extrinsèques """

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import glob

from tools.findPoints import proj_centers
from tools.util import coins_damier, clean_folders, draw_reprojection

# Data ========================================================================
dataPath="data/14_01_2021/"
# Data pour intrinsèque:
noFringePath=np.sort(glob.glob(os.path.join(dataPath,"zhang/max*.png")))
sgmfPath=np.sort(glob.glob(os.path.join(dataPath,"zhang/match*.png")))
# Data pour extrinsèque:
objpRT_path=os.path.join(dataPath,"RT-calib/objpts.txt")
imgpRT_path=os.path.join(dataPath,"RT-calib/imgpts.txt")
sgmfRT_path=os.path.join(dataPath,"RT-calib/match_00.png")

# Output:
outputPath=os.path.join(dataPath,"output2/")
imagesPath=os.path.join(outputPath,"detection/")
outputfile=os.path.join(outputPath,'erreur.txt')
# Créer/Vider les folder/file:
clean_folders([outputPath,imagesPath], ".png")
# ==============================================================================

# Paramètres ===================================================================
# Camera:
imageSize=(2464, 2056)
# Projecteur:
projSize=(1920,1200)
# Damier
points_per_row=10; points_per_colum=7
squaresize=10e-2
patternSize=(points_per_row, points_per_colum)
# ==============================================================================

# ==== LIRE LES POINTS =========================================================

# Damier -----------------------------------------------------------------------
# Listes de points
objectPoints=[]; imagePoints=[]; projPoints=[]
# Points 3d
objp = coins_damier(patternSize, squaresize)
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for i in range(len(noFringePath)):
    # Lire l'image
    color=cv.imread(noFringePath[i])
    gray=cv.cvtColor(color, cv.COLOR_BGR2GRAY)
    # Find corners
    ret, corners = cv.findChessboardCorners(gray, patternSize, None)
    if ret==True:
        imgp = cv.cornerSubPix(gray, corners, patternSize, (-1,-1), criteria)
        # Ajouter aux listes
        objectPoints.append(objp)
        imagePoints.append(imgp)
        # Points du projecteur
        projp = proj_centers(objp, imgp, projSize, sgmfPath[i])
        projPoints.append(projp)
        # Enregistrer les coins détectés - optionnel
        img = cv.drawChessboardCorners(color.copy(), patternSize, imgp, ret)
        cv.imwrite("{}corners_{}.png".format(imagesPath,i),img)
# ------------------------------------------------------------------------------

# Cibles -----------------------------------------------------------------------
# Lire les points
objpRT=np.genfromtxt(objpRT_path).astype(np.float32)
# offset x,y
objpRT[:,1]-=objpRT[2,1]
objpRT[:,0]-=objpRT[2,0]

imgpRT=np.genfromtxt(imgpRT_path).astype(np.float32)
imgpRT=imgpRT.reshape(24, 1, 2)
# Points du projecteur
projpRT = proj_centers(objpRT, imgpRT, projSize, sgmfRT_path)
# ------------------------------------------------------------------------------

# Choix de points pour la calibration intrinsèque:
objectPoints_=objectPoints.copy()
imagePoints_=imagePoints.copy()
projPoints_=projPoints.copy()
objectPoints_.append(objpRT)
imagePoints_.append(imgpRT)
projPoints_.append(projpRT)
#===============================================================================

#==== CALIBRATION INTRINSÈQUE ==================================================
# Camera -----------------------------------------------------------------------
retC, cameraMatrix, camDistCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors=cv.calibrateCameraExtended(objectPoints_, imagePoints_, imageSize, np.zeros((3,3)), np.zeros((1,4)))
# ------------------------------------------------------------------------------

# Projecteur -------------------------------------------------------------------
retP, projMatrix, projDistCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors=cv.calibrateCameraExtended(objectPoints_, projPoints_, projSize, np.zeros((3,3)), np.zeros((1,4)))
# ------------------------------------------------------------------------------
#===============================================================================

#==== CALIBRATION EXTRINSÈQUE ==================================================
retval, cameraMatrix, camDistCoeffs, projMatrix, projDistCoeffs, R, T, _, _, perViewErrors = cv.stereoCalibrateExtended([objpRT], [imgpRT], [projpRT], cameraMatrix, camDistCoeffs, projMatrix, projDistCoeffs, imageSize, None, None, flags=cv.CALIB_FIX_INTRINSIC)
#===============================================================================
color=cv.imread("data/14_01_2021/RT-calib/cornerSubPix.png")
img =draw_reprojection(color, outputPath, objpRT, imgpRT, cameraMatrix, camDistCoeffs, (4,6), 1)

#==== ENREGISTRER ==============================================================
# Enregistrer:
s = cv.FileStorage()
s.open('{}cam.xml'.format(outputPath), cv.FileStorage_WRITE)
s.write('K',cameraMatrix)
s.write('R', np.eye(3))
s.write('t', np.zeros(T.shape))
s.write('coeffs', camDistCoeffs)
s.write('imageSize', imageSize)
s.release()

# Enregistrer:
s = cv.FileStorage()
s.open('{}proj.xml'.format(outputPath), cv.FileStorage_WRITE)
s.write('K', projMatrix)
s.write('R', R)
s.write('t', T)
s.write('coeffs', projDistCoeffs)
s.write('imageSize', projSize)
s.release()
#===============================================================================
f=open(outputfile, 'w+'); f.close()
f=open(outputfile, 'a')
f.write('- Calibration Stéréo Zhang - \n \n')
f.write('Erreur de reprojection RMS caméra:\n')
f.write("{}\n".format(retC))
f.write('Erreur de reprojection RMS projecteur:\n')
f.write("{}\n".format(retP))
f.write('Erreur de reprojection RMS stéréo:\n')
f.write("{}\n".format(retval))
f.close()
