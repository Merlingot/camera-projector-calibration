"""calibration """

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import glob

from findPoints import camera_centers, proj_centers
from util import draw_reprojection, reprojection_err, formatage, outputClean

# Paramètres =============================================================
# Data:
SERIE="26_11_2020/party_mix"
# Camera:
imageSize = (2464, 2056)
# Projecteur:
projSize=(1920,1200)
# Damier
damier='square'
points_per_row=10; points_per_colum=7
spacing=10e-2
paperMargin=8.25e-2 # À trouver
patternSize=(points_per_row, points_per_colum)
patternSizeFull=(points_per_row*2, points_per_colum)
# ========================================================================

# Input:
dataPath="data/{}/".format(SERIE)
noFringePath=np.sort(glob.glob(os.path.join(dataPath,"max_*.png")))
sgmfPath=np.sort(glob.glob(os.path.join(dataPath,"match_*.png")))
# Output:
outputPath=os.path.join(dataPath,"output/")
# Points 3d et 2d:
pointsPath=outputPath
camPointsPath=os.path.join(pointsPath,"points_camera.txt")
projPointsPath=os.path.join(pointsPath,"points_camera.txt")
# Images de vérification
verifPath=outputPath
# Résultat de calibration
calibPath=outputPath
outputfile=os.path.join(calibPath,'calibration_double.txt')
# Créer/Vider les folder:
outputClean([verifPath, pointsPath, calibPath])
f=open(outputfile, 'w+'); f.close()

objectPoints0=[]; imagePoints0=[]; projPoints0=[]
objectPoints=[]; imagePoints=[]; projPoints=[]
for i in range(len(noFringePath)):

    # Zhang
    objp0, imgp0 = camera_centers(points_per_row, points_per_colum, paperMargin, spacing, None, noFringePath[i], verifPath, pointsPath, 'zhang', damier)
    n=objp0.shape[0]
    objectPoints0.append(objp0[:int(n/2),:])
    objectPoints0.append(objp0[int(n/2):,:])
    imagePoints0.append(imgp0[:int(n/2),:])
    imagePoints0.append(imgp0[int(n/2):,:])

    projp0 = proj_centers(objp0[:int(n/2),:], imgp0[:int(n/2),:], projSize, sgmfPath[i], pointsPath)
    projPoints0.append(projp0)
    projp0 = proj_centers(objp0[int(n/2):,:], imgp0[int(n/2):,:], projSize, sgmfPath[i], pointsPath)
    projPoints0.append(projp0)

    # Tsai
    objp, imgp = camera_centers(points_per_row, points_per_colum, paperMargin, spacing, None, noFringePath[i], verifPath, pointsPath, 'tsai', damier)
    objectPoints.append(objp)
    imagePoints.append(imgp)

    projp = proj_centers(objp, imgp, projSize, sgmfPath[i], pointsPath)
    projPoints.append(projp)

# Camera ----------------------------------------------
retval, cameraMatrix0, camDistCoeffs0, _, _, _, _, perViewErrors0=cv.calibrateCameraExtended(objectPoints0, imagePoints0, imageSize, np.zeros((3,3)), np.zeros((1,4)))

retval, cameraMatrix, camDistCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors=cv.calibrateCameraExtended(objectPoints, imagePoints, imageSize, cameraMatrix0, camDistCoeffs0,flags=cv.CALIB_USE_INTRINSIC_GUESS)
# --------------------------------------------------------

# Projecteur ----------------------------------------------
retval, projMatrix0, projDistCoeffs0, _, _, _, _, perViewErrors0=cv.calibrateCameraExtended(objectPoints0, projPoints0, projSize, np.zeros((3,3)), np.zeros((1,4)))

retval, projMatrix, projDistCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors=cv.calibrateCameraExtended(objectPoints, projPoints, projSize, projMatrix0, projDistCoeffs0, flags=cv.CALIB_USE_INTRINSIC_GUESS)
# --------------------------------------------------------


# # stereoCalibrate:
retval, cameraMatrix0, camDistCoeffs0, projMatrix0, projDistCoeffs0, R0, T0, E0, F0, perViewErrors0 = cv.stereoCalibrateExtended(objectPoints0, imagePoints0, projPoints0, cameraMatrix0, camDistCoeffs0, projMatrix0, projDistCoeffs0, imageSize, np.zeros((3,3)), np.zeros(3))

f=open(outputfile, 'a')
f.write('- Calibration Stéréo Zhang - \n \n')
f.write('Erreur de reprojection RMS:\n')
f.write("{}\n".format(np.mean(perViewErrors0)))
f.write('Matrice de rotation:\n')
f.write("{}\n".format(R0))
f.write('Vecteur translation:\n')
f.write("{}\n".format(T0))
f.write('Distance euclidienne caméra-projecteur:\n')
f.write("{}\n\n".format(np.linalg.norm(T0)))
f.close()

retval, cameraMatrix, camDistCoeffs, projMatrix, projDistCoeffs, R, T, E, F, perViewErrors = cv.stereoCalibrateExtended(objectPoints, imagePoints, projPoints, cameraMatrix, camDistCoeffs, projMatrix, projDistCoeffs, imageSize, np.zeros((3,3)), np.zeros(3))

f=open(outputfile, 'a')
f.write('- Calibration Stéréo Tsai - \n \n')
f.write('Erreur de reprojection RMS:\n')
f.write("{}\n".format(np.mean(perViewErrors)))
f.write('Matrice de rotation:\n')
f.write("{}\n".format(R))
f.write('Vecteur translation:\n')
f.write("{}\n".format(T))
f.write('Distance euclidienne caméra-projecteur:\n')
f.write("{}\n\n".format(np.linalg.norm(T)))
f.close()

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
