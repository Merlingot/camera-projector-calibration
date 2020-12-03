"""calibration """

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import glob

from tools.findPoints import camera_centers, proj_centers, get_objp
from tools.util import draw_reprojection, reprojection_err, formatage, outputClean

# Paramètres =============================================================
# Data:
SERIE="03_12_2020/"
# Camera:
imageSize = (2464, 2056)
# Projecteur:
projSize=(1920,1200)
# Damier
damier='simple'
motif='square'
points_per_row=10; points_per_colum=7
spacing=10e-2
paperMargin=0 # À trouver
patternSize=(points_per_row, points_per_colum)
patternSizeFull=(points_per_row*2, points_per_colum)
# ========================================================================

# Input:
dataPath="data/{}/".format(SERIE)
noFringePath=np.sort(glob.glob(os.path.join(dataPath,"max_*.png")))
sgmfPath=np.sort(glob.glob(os.path.join(dataPath,"match_*.png")))

# Output:
outputPath=os.path.join(dataPath,"output/")
imagesPath=os.path.join(outputPath,"images/")
outputfile=os.path.join(outputPath,'calibration.txt')
# Créer/Vider les folder:
outputClean([outputPath,imagesPath])
f=open(outputfile, 'w+'); f.close()


# Obtenir les points -----------------------------------
objectPoints=[]; imagePoints=[]; projPoints=[]
for i in range(len(noFringePath)):

    # Lire l'image
    color=cv.imread(noFringePath[i])
    gray=cv.cvtColor(color , cv.COLOR_BGR2GRAY)

    # Points 3d
    _, objp = get_objp(points_per_row, points_per_colum, paperMargin, spacing, None, None, motif, damier)
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Find corners
    ret, corners = cv.findChessboardCorners(gray, patternSize, None)
    if ret==True:
        imgp = cv.cornerSubPix(gray, corners, patternSize, (-1,-1), criteria)
        img = cv.drawChessboardCorners(color.copy(), patternSize, imgp, ret)
        cv.imwrite("{}centers_{}.png".format(imagesPath,i),img)
        objectPoints.append(objp)
        imagePoints.append(imgp)

        projp = proj_centers(objp, imgp, projSize, sgmfPath[i], outputPath)
        projPoints.append(projp)
# --------------------------------------------------------

# Camera ----------------------------------------------
retval, cameraMatrix, camDistCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors=cv.calibrateCameraExtended(objectPoints, imagePoints, imageSize, np.zeros((3,3)), np.zeros((1,4)))

# Enlever les outliers et recalibrer:
indices=np.indices(perViewErrors.shape)[0]
indexes=indices[perViewErrors>retval*2]
for i in indexes:
    objectPoints.pop(i)
    imagePoints.pop(i)
    projPoints.pop(i)

retC, cameraMatrix, camDistCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors=cv.calibrateCameraExtended(objectPoints, imagePoints, imageSize, np.zeros((3,3)), np.zeros((1,4)))
# --------------------------------------------------------

# Projecteur ----------------------------------------------
retP, projMatrix, projDistCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors=cv.calibrateCameraExtended(objectPoints, projPoints, projSize, np.zeros((3,3)), np.zeros((1,4)))
# --------------------------------------------------------

retval, cameraMatrix, camDistCoeffs, projMatrix, projDistCoeffs, R, T, E, F = cv.stereoCalibrate(objectPoints, imagePoints, projPoints, cameraMatrix, camDistCoeffs, projMatrix, projDistCoeffs, imageSize, flags=cv.CALIB_FIX_INTRINSIC)

f=open(outputfile, 'a')
f.write('- Calibration Stéréo Zhang - \n \n')
f.write('Erreur de reprojection RMS:\n')
f.write("{}\n".format(retval))
f.write('Matrice de rotation:\n')
f.write("{}\n".format(R))
f.write('Vecteur translation:\n')
f.write("{}\n".format(T))
f.write('Distance euclidienne caméra-projecteur (m):\n')
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
