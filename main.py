""" Tsai calibration """

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

from findPoints import camera_centers, proj_centers
from tsai_stage1 import calibrate
from tsai_stage2 import nonLinearSearch
from util import draw_reprojection, reprojection_err, formatage, outputClean

# Paramètres =============================================================
# Data:
SERIE="13_11_2020/louis"
# Camera:
imageSize = (2464, 2056)
camPixelSize = (3.45e-6, 3.45e-6)
# Projecteur:
projSize=(1920,1200)
projPixelSize = (0.269e-3, 0.269e-3)
# Damier
points_per_row=8; points_per_colum=8
circleDiameter=10e-2
circleSpacing=20e-2
paperMargin=20e-2 # À trouver
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
outputfile=os.path.join(calibPath,'calibration.txt')
# Créer/Vider les folder:
outputClean([verifPath, pointsPath, calibPath])


# CAMERA
# Trouver les cercles:
objp, imgp = camera_centers(points_per_row, points_per_colum, paperMargin, circleSpacing, circleDiameter, noFringePath, verifPath, pointsPath )
# Paramètre linéaires
data, params, R, T, f, sx = calibrate(camPointsPath, imageSize, camPixelSize)
# Paramètres non linéaires
f_, Tz_, k1_, k2_= nonLinearSearch(data.world, data.realImage, R, T, f, sx)
# Reprojection:
imagePoints, objectPoints, rvec, tvec, cameraMatrix, camDistCoeffs=formatage(data.n, data.world, data.computerImage, params.Cx, params.Cy, params.dx, params.dy, R, T, sx, f_, Tz_, k1_, k2_)
projectedPoints, _ = cv.projectPoints(objectPoints, rvec, tvec, cameraMatrix, camDistCoeffs)
err = reprojection_err(imagePoints, projectedPoints)
# Images pour le fun:
draw_reprojection(cv.imread(noFringePath), calibPath, objectPoints, imagePoints, cameraMatrix, camDistCoeffs, patternSizeFull)
undistort = cv.undistort(cv.imread(noFringePath), cameraMatrix, camDistCoeffs)
cv.imwrite( '{}undistort.png'.format(outputPath), undistort )

f=open(outputfile, 'w+')
f.write('- Camera -\n\n')
f.write('Erreur de reprojection RMS - Après optimisation:\n')
f.write("{}\n".format(err))
f.write('Matrice de rotation:\n')
f.write("{}\n".format(R))
f.write('Vecteur translation:\n')
f.write("{}\n".format(tvec))
f.write('Matrice paramètres intrinsèque:\n')
f.write("{}\n".format(cameraMatrix))
f.write('Coefficients de distorsion:\n')
f.write('{}\n\n'.format(camDistCoeffs))
f.close()

# Projecteur
# Trouver les cercles:
# imgp_undistort = cv.undistortPoints(imgp, cameraMatrix, camDistCoeffs)
proj_centers(objp, imgp, projSize, sgmfPath, pointsPath)
# Paramètre linéaires
data, params, R, T, f, sx = calibrate(projPointsPath, projSize, projPixelSize)
# Paramètres non linéaires
f_, Tz_, k1_, k2_= nonLinearSearch(data.world, data.realImage, R, T, f, sx)
# Reprojection:
projectorPoints, objectPoints, rvec, tvec, projMatrix, projDistCoeffs=formatage(data.n, data.world, data.computerImage, params.Cx, params.Cy, params.dx, params.dy, R, T, sx, f_, Tz_, k1_, k2_)
projectedPoints, _ = cv.projectPoints(objectPoints, rvec, tvec, projMatrix, projDistCoeffs)
err = reprojection_err(projectorPoints, projectedPoints)

f=open(outputfile, 'a')
f.write('- Projecteur -\n\n')
f.write('Erreur de reprojection RMS - Après optimisation:\n')
f.write("{}\n".format(err))
f.write('Matrice de rotation:\n')
f.write("{}\n".format(R))
f.write('Vecteur translation:\n')
f.write("{}\n".format(tvec))
f.write('Matrice paramètres intrinsèque:\n')
f.write("{}\n".format(projMatrix))
f.write('Coefficients de distorsion:\n')
f.write('{}\n\n'.format(projDistCoeffs))
f.close()

# # stereoCalibrate:
retval, cameraMatrix, camDistCoeffs, projMatrix, projDistCoeffs, R, T, E, F, perViewErrors = cv.stereoCalibrateExtended([objectPoints.astype(np.float32)], [imagePoints.astype(np.float32)], [projectorPoints.astype(np.float32)], cameraMatrix, camDistCoeffs, projMatrix, projDistCoeffs, imageSize, np.zeros((3,3)), np.zeros(3))

f=open(outputfile, 'a')
f.write('- Calibration Stéréo - \n \n')
f.write('Erreur de reprojection RMS:\n')
f.write("{}\n".format(np.sum(perViewErrors)/perViewErrors.shape[0]))
f.write('Matrice de rotation:\n')
f.write("{}\n".format(R))
f.write('Vecteur translation:\n')
f.write("{}\n".format(T))
f.write('Distance euclidienne caméra-projecteur:\n')
f.write("{}\n\n".format(np.linalg.norm(T)))
f.close()


# StereoRectify
R1, R2, P1, P2, Q, validPixROI1, validPixROI2=cv.stereoRectify(	cameraMatrix, camDistCoeffs, projMatrix, projDistCoeffs, imageSize, R, T)

f=open(outputfile, 'a')
f.write('- Rectification stéréo - \n\n')
f.write('Matrice de rectification, Camera:\n')
f.write("{}\n".format(R1))
f.write('Matrice de rectification, Projecteur:\n')
f.write("{}\n".format(R2))
f.write('Matrice de projection, Camera:\n')
f.write("{}\n".format(P1))
f.write('Matrice de projection, Projecteur:\n')
f.write("{}\n".format(P2))
f.write('Matrice Q:\n')
f.write("{}".format(Q))
f.close()










#
