""" calibration Zhang pour intrinsèques et avec les cibles pour extrinsèques """

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import glob

from tools.findPoints import proj_centers
from tools.util import coins_damier, clean_folders

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
outputPath=os.path.join(dataPath,"output/")
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


#==== CALIBRATION INTRINSÈQUE ==================================================
# Listes de points
objectPoints=[]; imagePoints=[]; projPoints=[]
# Points 3d
objp = coins_damier(patternSize, squaresize)
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Obtenir les points -----------------------------------------------------------
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

# Camera -----------------------------------------------------------------------
retC, cameraMatrix, camDistCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors=cv.calibrateCameraExtended(objectPoints, imagePoints, imageSize, np.zeros((3,3)), np.zeros((1,4)))

# Enlever les outliers et recalibrer:
indices=np.indices(perViewErrors.shape)[0]
indexes=indices[perViewErrors>retC*2]
if len(indexes) > 0: # Si au moins 1 outlier, recalibrer
    for i in indexes:
        objectPoints.pop(i)
        imagePoints.pop(i)
        projPoints.pop(i)
    retC, cameraMatrix, camDistCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors=cv.calibrateCameraExtended(objectPoints, imagePoints, imageSize, np.zeros((3,3)), np.zeros((1,4)))
# ------------------------------------------------------------------------------

# Projecteur -------------------------------------------------------------------
retP, projMatrix, projDistCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors=cv.calibrateCameraExtended(objectPoints, projPoints, projSize, np.zeros((3,3)), np.zeros((1,4)))
# ------------------------------------------------------------------------------
#===============================================================================


#==== CALIBRATION EXTRINSÈQUE ==================================================
# Lire les points
objpRT=np.genfromtxt(objpRT_path).astype(np.float32)
imgpRT=np.genfromtxt(imgpRT_path).astype(np.float32)
imgpRT=imgpRT.reshape(24, 1, 2)
# Points du projecteur
projpRT = proj_centers(objpRT, imgpRT, projSize, sgmfRT_path)
plt.plot(objpRT[:,0],objpRT[:,1], 'o-')
plt.plot(projpRT[:,0,0], projpRT[:,0,1], 'o-')

retval, cameraMatrix, camDistCoeffs, projMatrix, projDistCoeffs, R, T, _, _ = cv.stereoCalibrate([objpRT], [imgpRT], [projpRT], cameraMatrix, camDistCoeffs, projMatrix, projDistCoeffs, imageSize, flags=cv.CALIB_FIX_INTRINSIC)
#===============================================================================

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
# f=open(outputfile, 'w+'); f.close()
# f=open(outputfile, 'a')
# f.write('- Calibration Stéréo Zhang - \n \n')
# f.write('Erreur de reprojection RMS caméra:\n')
# f.write("{}\n".format(retC))
# f.write('Erreur de reprojection RMS projecteur:\n')
# f.write("{}\n".format(retP))
# f.write('Erreur de reprojection RMS stéréo:\n')
# f.write("{}\n".format(retval))
# f.close()
