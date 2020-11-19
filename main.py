"""calibration """

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

from findPoints import camera_centers, proj_centers
from util import draw_reprojection, reprojection_err, formatage, outputClean

# Paramètres =============================================================
# Data:
SERIE="13_11_2020/louis"
# Camera:
imageSize = (2464, 2056)
# Projecteur:
projSize=(1920,1200)
# Damier
points_per_row=8; points_per_colum=8
circleDiameter=10e-2
circleSpacing=20e-2
paperMargin=20e-2 # À trouver
patternSize=(points_per_colum,points_per_row)
patternSizeFull=(points_per_colum,points_per_row*2)
# ========================================================================

# Input:
dataPath="data/{}/".format(SERIE)
noFringePath=os.path.join(dataPath,"nofringe/noFringe.png")
sgmfPath=os.path.join(dataPath, "cam_match.png")
# Output:
outputPath=os.path.join(dataPath,"output/")
# Points 3d et 2d:
pointsPath=os.path.join(outputPath,"points/")
camPointsPath=os.path.join(pointsPath,"points_camera.txt")
projPointsPath=os.path.join(pointsPath,"points_camera.txt")
# Images de vérification
verifPath=os.path.join(outputPath,"detection/")
# Résultat de calibration
calibPath=os.path.join(outputPath,"calibration/")
outputfile=os.path.join(calibPath,'calibration_double.txt')
# Créer/Vider les folder:
outputClean([verifPath, pointsPath, calibPath])
f=open(outputfile, 'w+'); f.close()


# CAMERA ----------------------------------------------
# Premiere estimation avec 2 vues coplanaires:
objp0, imgp0 = camera_centers(points_per_row, points_per_colum, paperMargin, circleSpacing, circleDiameter, noFringePath, verifPath, pointsPath, 'zhang')
n=objp0.shape[0]
objectPoints=[objp0[:int(n/2),:],objp0[int(n/2):,:]]
imagePoints=[imgp0[:int(n/2),:],imgp0[int(n/2):,:]]
retval, cameraMatrix0, _, _, _, _, _, perViewErrors0=cv.calibrateCameraExtended(objectPoints, imagePoints, imageSize, np.zeros((3,3)), np.zeros((1,4)))

# Deuxième estimation avec 1 vue non coplanaire
objp, imgp = camera_centers(points_per_row, points_per_colum, paperMargin, circleSpacing, circleDiameter, noFringePath, verifPath, pointsPath, 'tsai')
retval, cameraMatrix, camDistCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors=cv.calibrateCameraExtended([objp], [imgp], imageSize, cameraMatrix0, np.zeros((1,4)),flags=cv.CALIB_USE_INTRINSIC_GUESS)

# Image pour le fun:
draw_reprojection(cv.imread(noFringePath), calibPath, objp, imgp, cameraMatrix, camDistCoeffs, patternSizeFull)

f=open(outputfile, 'a')
f.write('- Camera double méthode -\n\n')
f.write('Erreur de reprojection RMS:\n')
f.write("{}\n".format(perViewErrors0))
f.write('Erreur de reprojection RMS apres recalibration:\n')
f.write("{}\n".format(perViewErrors))
f.write('Matrice de rotation:\n')
f.write("{}\n".format(rvecs))
f.write('Vecteur translation:\n')
f.write("{}\n".format(tvecs))
f.write('Matrice paramètres intrinsèque:\n')
f.write("{}\n".format(cameraMatrix))
f.write('Coefficients de distorsion:\n')
f.write('{}\n\n'.format(camDistCoeffs))
f.close()
# --------------------------------------------------------

# Projecteur ----------------------------------------------

# Premiere estimation avec 2 vues coplanaire
projp0 = proj_centers(objp0, imgp0, projSize, sgmfPath, pointsPath)
n=objp.shape[0]
objectPoints=[objp0[:int(n/2),:],objp0[int(n/2):,:]]
projPoints=[projp0[:int(n/2),:],projp0[int(n/2):,:]]
retval, projMatrix0, _, _, _, _, _, perViewErrors0=cv.calibrateCameraExtended(objectPoints, projPoints, projSize, np.zeros((3,3)), np.zeros((1,4)))

# Deuxième estimation avec 1 vue non coplanaire
projp = proj_centers(objp, imgp, projSize, sgmfPath, pointsPath)
retval, projMatrix, projDistCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors=cv.calibrateCameraExtended([objp], [projp], projSize, projMatrix0, np.zeros((1,4)), flags=cv.CALIB_USE_INTRINSIC_GUESS)

# Image pour le fun:
# projectedPoints, _ = cv.projectPoints(objp, rvecs[0], tvecs[0], projMatrix, projDistCoeffs)
# plt.figure()
# plt.plot(projp[:,0,0], projp[:,0,1], 'ro')
# plt.plot(projectedPoints[:,0,0], projectedPoints[:,0,1], 'bo')
# plt.show()

f=open(outputfile, 'a')
f.write('- Projecteur double méthode -\n\n')
f.write('Erreur de reprojection RMS:\n')
f.write("{}\n".format(perViewErrors0))
f.write('Erreur de reprojection RMS apres recalibration:\n')
f.write("{}\n".format(perViewErrors))
f.write('Matrice de rotation:\n')
f.write("{}\n".format(rvecs))
f.write('Vecteur translation:\n')
f.write("{}\n".format(tvecs))
f.write('Matrice paramètres intrinsèque:\n')
f.write("{}\n".format(projMatrix))
f.write('Coefficients de distorsion:\n')
f.write('{}\n\n'.format(projDistCoeffs))
f.close()
# --------------------------------------------------------


# # stereoCalibrate:
retval, cameraMatrix, camDistCoeffs, projMatrix, projDistCoeffs, R, T, E, F, perViewErrors = cv.stereoCalibrateExtended([objp], [imgp], [projp], cameraMatrix, camDistCoeffs, projMatrix, projDistCoeffs, imageSize, np.zeros((3,3)), np.zeros(3))

f=open(outputfile, 'a')
f.write('- Calibration Stéréo - \n \n')
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
s.write('R', R)
s.write('t', T)
s.write('norme_t', np.linalg.norm(T))
s.write('coeffs', camDistCoeffs)
s.write('imageSize', imageSize)
s.release()


retval, projMatrix, projDistCoeffs, cameraMatrix, camDistCoeffs, R, T, E, F, perViewErrors = cv.stereoCalibrateExtended([objp], [projp], [imgp], projMatrix, projDistCoeffs, cameraMatrix, camDistCoeffs, projSize, np.zeros((3,3)), np.zeros(3))

# Enregistrer:
s = cv.FileStorage()
s.open('{}proj.xml'.format(outputPath), cv.FileStorage_WRITE)
s.write('K', projMatrix)
s.write('R', R)
s.write('t', T)
s.write('norme_t', np.linalg.norm(T))
s.write('coeffs', projDistCoeffs)
s.write('imageSize', projSize)
s.release()
