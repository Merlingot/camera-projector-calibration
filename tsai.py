""" Tsai calibration """

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from calibration import calibrate


# Paramètre linéaires
pointsPath="data/13_11_2020/serie_gp_1/points/points_camera.txt"
imageShape = (2464, 2056)
pixelSize = (3.45e-6, 3.45e-6)
data, params, R, T, f, sx = calibrate(pointsPath, imageShape, pixelSize)
# Paramètres non linéaires
from nonLinearSearch import nonLinearSearch
f_, Tz_, k1_, k2_= nonLinearSearch(data, params, R, T, f, sx)

# Reprojection:

def intrinsicMatrix(f, Cx, Cy, dx, dy):
    mx = 1/dx; my=1/dy
    return np.array([ [f*mx, 0, Cx],[0,f*my,Cy],[0,0,1] ])

def formatage(data, params, R, T, sx, f_, Tz_, k1_, k2_):
    # preparer les données pour project points
    objectPoints = np.zeros((data.n,3))
    imagePoints = np.zeros((data.n,1,2))
    for i in range(data.n):
        objectPoints[i,:] = np.array([data.world[0][i], data.world[1][i], data.world[2][i]])
        imagePoints[i,0,:] = np.array([data.computerImage[0][i], data.computerImage[1][i]])
    # Rigid transform:
    rvec, _ = cv.Rodrigues(R)
    tvec = np.array([T[0],T[1],Tz_])
    # Intrinsic Matrix :
    cameraMatrix = intrinsicMatrix(f_, params.Cx, params.Cy, params.dx, params.dy)
    # Distortion coefficients
    distCoeffs = np.array([k1_, k2_,0,0])
    return imagePoints, objectPoints, rvec, tvec, cameraMatrix, distCoeffs

def reprojection_err(imagePoints, projectedPoints):
    err = np.sum(( imagePoints[:,0,0] -  projectedPoints[:,0,0] )**2 ) + np.sum( ( imagePoints[:,0,1] -  projectedPoints[:,0,1] )**2)
    n=imagePoints.shape[0]
    return np.sqrt(err/n)


imagePoints, objectPoints, rvec, tvec, cameraMatrix, distCoeffs=formatage(data, params, R, T, sx, f, T[2], 0, 0)
projectedPoints, _ = cv.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs=None)

err = reprojection_err(imagePoints, projectedPoints)
print('Mean reprojection error (pixel) - First estimate')
print(err)
plt.plot(imagePoints[:,0,0], imagePoints[:,0,1], 'r.', projectedPoints[:,0,0], projectedPoints[:,0,1], 'b.')
plt.show()

imagePoints, objectPoints, rvec, tvec, cameraMatrix, distCoeffs=formatage(data, params, R, T, sx, f_, Tz_, k1_, k2_)
projectedPoints, _ = cv.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs)

err = reprojection_err(imagePoints, projectedPoints)
print('Mean reprojection error (pixel) - After non linear search')
print(err)
plt.plot(imagePoints[:,0,0], imagePoints[:,0,1], 'r.', projectedPoints[:,0,0], projectedPoints[:,0,1], 'b.')
plt.show()
