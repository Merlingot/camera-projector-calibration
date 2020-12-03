""" Tsai calibration """

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

from tools.findPoints import camera_centers, proj_centers
from tools.tsai_stage1 import calibrate
from tools.tsai_stage2 import nonLinearSearch
from tool.util import draw_reprojection, reprojection_err, formatage, outputClean

# Paramètres =============================================================
# Data:
SERIE="26_11_2020/5500mm"
# Camera:
imageSize = (2464, 2056)
camPixelSize=(3.45e-6, 3.45e-6)
# Projecteur:
projSize=(1920,1200)
# Damier
damier='double'
motif='square'
points_per_row=10; points_per_colum=7
spacing=10e-2
paperMargin=8.25e-2 # À trouver
patternSize=(points_per_row, points_per_colum)
patternSizeFull=(points_per_row*2, points_per_colum)
# ========================================================================
# Input
dataPath="data/{}/".format(SERIE)
noFringePath=os.path.join(dataPath,"max_00.png")
sgmfPath=os.path.join(dataPath, "match_00.png")
# Output
outputPath=os.path.join(dataPath,"output/")
camPointsPath=os.path.join(outputPath,"points_camera.txt")
# -> Résultat de calibration
outputfile=os.path.join(outputPath,'calibration_tsai.txt')
# Créer/Vider les folder
outputClean([outputPath])
f=open(outputfile, 'w+'); f.close()

# CAMERA
objp, imgp = camera_centers(points_per_row, points_per_colum, paperMargin, spacing, None, noFringePath, outputPath, 'tsai', motif, outputPath)
# Paramètre linéaires
data, params, R, T, f, sx = calibrate(camPointsPath, imageSize, camPixelSize)
# Paramètres non linéaires
f_, Tz_, k1_, k2_= nonLinearSearch(data.world, data.realImage, R, T, f)
# Reprojection:
imagePoints, objectPoints, rvec, tvec, cameraMatrix, camDistCoeffs=formatage(data.n, data.world, data.computerImage, params.Cx, params.Cy, params.dx, params.dy, R, T, sx, f_, Tz_, k1_, k2_)
pts, _ = cv.projectPoints(objectPoints, rvec, tvec, cameraMatrix, camDistCoeffs)
err = reprojection_err(imgp, pts)
# Image pour le fun:
_=draw_reprojection(cv.imread(noFringePath), outputPath, objp, imgp, cameraMatrix, camDistCoeffs, patternSizeFull)

f=open(outputfile, 'a')
f.write('- Camera Tsai -\n\n')
f.write('Erreur de reprojection RMS:\n')
f.write("{}\n".format(err))
f.write('Matrice de rotation:\n')
f.write("{}\n".format(R))
f.write('Vecteur translation:\n')
f.write("{}\n".format(T))
f.write('Matrice paramètres intrinsèque:\n')
f.write("{}\n".format(cameraMatrix))
f.write('Coefficients de distorsion:\n')
f.write('{}\n\n'.format(camDistCoeffs))
f.close()
# --------

# Projecteur ----------------------------------------------
# Premiere estimation avec 2 vues coplanaire
_, objp0 =get_objp(points_per_row, points_per_colum, paperMargin, spacing, None, 'zhang', motif, damier)
projp0 = proj_centers(objp0, imgp, projSize, sgmfPath, outputPath)
n=objp.shape[0]
objectPoints=[objp0[:int(n/2),:],objp0[int(n/2):,:]]
projPoints=[projp0[:int(n/2),:],projp0[int(n/2):,:]]
retval, projMatrix0, _, _, _, _, _, perViewErrors0=cv.calibrateCameraExtended(objectPoints, projPoints, projSize, np.zeros((3,3)), np.zeros((1,4)))

# Deuxième estimation avec 1 vue non coplanaire
projp = proj_centers(objp, imgp, projSize, sgmfPath, outputPath)
retval, projMatrix, projDistCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors=cv.calibrateCameraExtended([objp], [projp], projSize, projMatrix0, np.zeros((1,4)), flags=cv.CALIB_USE_INTRINSIC_GUESS)

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
retval, cameraMatrix, camDistCoeffs, projMatrix, projDistCoeffs, R, T, E, F, perViewErrors = cv.stereoCalibrateExtended([objp],  [imgp], [projp], cameraMatrix, camDistCoeffs, projMatrix, projDistCoeffs, imageSize, np.zeros((3,3)), np.zeros(3))

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








#
