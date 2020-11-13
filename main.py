""" Fichier pour détecter les centres des cercles et écrire les coordonées dans un fichier .txt"""
import os
import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt

# Personal imports :
import sgmf
from detectCenters import detect_centers
from circlesgrid import getAsymCirclesObjPoints
import util

#Résolution du projecteur en pixel
projSize=(800,600)


# Data:
SERIE="serie_1"
dataPath="data/{}/".format(SERIE)
noFringePath=os.path.join(dataPath, "nofringe/noFringe.png")
sgmfPath=os.path.join(dataPath, "cam_match.png")

verifPath=os.path.join(dataPath,"verif/")
pointsPath=os.path.join(dataPath,"points/")
output_paths=[verifPath, pointsPath]
util.outputClean(output_paths)


# Lire l'image
color=cv.imread(noFringePath)
gray = cv.cvtColor(color , cv.COLOR_BGR2GRAY)
# Résolution caméra:
imageSize=gray.shape

# Damier
points_per_row=4; points_per_colum=11
patternSize = (points_per_row, points_per_colum)
realPatternSize=(2*points_per_row, points_per_colum)
circleDiameter=1.5e-2
circleSpacing=2e-2
paperMargin = 3e-2 #À trouver
offset=paperMargin+(circleSpacing+circleDiameter)/2
# Points 3d
objpR = getAsymCirclesObjPoints(points_per_colum, points_per_row, circleDiameter+circleSpacing, offset, 0, "xy")
objpL = getAsymCirclesObjPoints(points_per_colum, points_per_row, circleDiameter+circleSpacing, offset, 0, "yz")
objp=np.concatenate((objpR, objpL))

# Vérification des points 3D
# fig, ax = plt.subplots()
# plt.title('Coté droit')
# plt.xlabel('x (m)')
# plt.ylabel('y (m)')
# plt.xlim(0, np.max(objpR[:,0])*1.2)
# plt.plot(objpR[:,0], objpR[:,1],'o-')
# plt.plot(objpR[:,0][0], objpR[:,1][0],'o', color='C1')
# plt.savefig('{}damier_droit.png'.format(verifPath), format="png")
#
# fig, ax = plt.subplots()
# plt.plot(objpL[:,2], objpL[:,1],'o-')
# plt.plot(objpL[:,2][0], objpL[:,1][0],'o', color='C1')
# plt.title('Coté gauche')
# plt.xlabel('z (m)')
# plt.ylabel('y (m)')
# ax.set_xlim(np.max(objpL[:,2])*1.2, 0)
# plt.savefig('{}damier_gauche.png'.format(verifPath), format="png")
#
# fig, ax = plt.subplots()
# ax.set_title('Damier complet vue de haut')
# ax.set_xlabel('x (m)')
# ax.set_ylabel('z (m)')
# ax.plot(objp[:,0], objp[:,2],'o')
# plt.savefig('{}damier_vue_haut.png'.format(verifPath), format="png")



# Détecter les coins  ----------------------------------
imagePoints=detect_centers(patternSize, objp, color, gray, verifPath, pointsPath)
objectPoints=[objpR.astype(np.float32), objpL.astype(np.float32)]
# Format de imgPoints et objPoints (pour la calibration) À CORRIGER
objPoints=[objpR.astype(np.float32)]
imgPoints=[imagePoints[0]]

# plt.figure()
# plt.title('vérification cercles')
# plt.imshow(gray)
# plt.plot( imgPoints[0][:,0,0], imgPoints[0][:,0,1] )


# Calibration de la caméra (estimation)  ----------------------------------
ret, cameraMatrix, camDistCoeffs, _, _ = cv.calibrateCamera(objPoints, imgPoints, gray.shape[::-1],None,None)


def draw(img, origin, imgpts):
    img = cv.line(img, tuple(origin[0].ravel()), tuple(imgpts[0].ravel()), (255,0,0), 5) #X
    img = cv.line(img, tuple(origin[0].ravel()), tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, tuple(origin[0].ravel()), tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

# Vérification de la calibration de la caméra en reprojection:
# Montrer axes
axis = np.float32([[3,0,0], [0,3,0], [0,0,3]]).reshape(-1,3)*1e-1
ret, rvecs, tvecs = cv.solvePnP(objPoints[0], imgPoints[0], cameraMatrix, camDistCoeffs)
axisProj, jac = cv.projectPoints(axis, rvecs, tvecs, cameraMatrix, camDistCoeffs)
origin = np.float32([[0,0,0]]).reshape(-1,1)
originProj , jac = cv.projectPoints(origin, rvecs, tvecs, cameraMatrix, camDistCoeffs)
img = draw(color.copy(), originProj[0], axisProj)
cv.imwrite('{}reprojection_axes.png'.format(verifPath), img)
# À droite
pts = objpR.astype(np.float32)
ptsProj, jac = cv.projectPoints(pts, rvecs, tvecs, cameraMatrix, camDistCoeffs)
img = cv.drawChessboardCorners(color.copy(), patternSize, ptsProj, 1)
cv.imwrite('{}reprojection_droite.png'.format(verifPath), img)
# À gauche
pts = objpL.astype(np.float32)
ptsProj, jac = cv.projectPoints(pts, rvecs, tvecs, cameraMatrix, camDistCoeffs)
img = cv.drawChessboardCorners(color.copy(), patternSize, ptsProj, 1)
cv.imwrite('{}reprojection_gauche.png'.format(verifPath), img)



# Calibration du projecteur -------------------------
def get_projPoints(sgmf, imagePoints):
    projPoints = imagePoints.copy()
    for i in range(imagePoints.shape[0]):
        circle=imagePoints[i][0] #Format bizare
        # sans transformation affine:
        p=sgmf.get_value(circle)
        projPoints[i,0,:]=np.array([ np.array([p[0],p[1]]).astype(np.float32) ]) #Format bizare
    return [projPoints]

# Extraction des coordonnées des points dans le plan image du projecteur
SGMF=sgmf.sgmf(sgmfPath, projSize, shadowMaskName=None)
projPoints=get_projPoints(SGMF, imgPoints[0])
# Vérification
plt.figure()
plt.title('Vérification points images')
plt.imshow(SGMF.channelX)
plt.plot(imagePoints[0][:,0,0],imagePoints[0][:,0,1], 'o')
plt.plot(imagePoints[1][:,0,0],imagePoints[1][:,0,1], 'o')
plt.savefig('{}superposionSGMF.png'.format(verifPath))

# Calibration du projecteur
ret, projMatrix, projDistCoeffs, prvecs, ptvecs = cv.calibrateCamera(objPoints, projPoints, gray.shape[::-1],None,None)

# Calibration stéréo  ----------------------------------

projectorPoints=[ projPoints[0], get_projPoints(SGMF, imagePoints[1])[0] ]
plt.figure()
plt.title('Points plan image projecteur trouvés avec sgmf')
plt.plot(projectorPoints[0][:,0,0], projectorPoints[0][:,0,1], 'o')
plt.plot(projectorPoints[1][:,0,0], projectorPoints[1][:,0,1], 'o')
plt.savefig('{}points_projecteur.png'.format(verifPath))

retval, cameraMatrix, camDistCoeffs, projMatrix, projDistCoeffs, R, T, E, F, perViewErrors = cv.stereoCalibrateExtended(objectPoints, imagePoints, projectorPoints, cameraMatrix, camDistCoeffs, projMatrix, projDistCoeffs, imageSize, np.zeros((3,3)), np.zeros(3))










































# def get_projPoints(sgmf, imagePoints):
#     projPoints = imagePoints.copy()
#     for i in range(imagePoints.shape[0]):
#         circle=imagePoints[i][0] #Format bizare
#         # srcPoints, dstPoints = get_entourage(sgmf,circle) #coordonées 2D (non homogène)
#         # mat,_ = cv.estimateAffine2D(srcPoints, dstPoints)
#         # A = np.array([[mat[0,0],mat[0,1]],[mat[1,0],mat[1,1]]]); b=mat[:,2]
#         # # print(circle)
#         # p=A@circle + b
#         # projPoints[i,0,:] = np.array([p.astype(np.float32)])
#
#         # Autre méthode:
#         p=sgmf.get_value(circle)
#         projPoints[i,0,:]=np.array([ np.array([ p[0],p[1]]).astype(np.float32) ]) #Format bizare
#
#     return [projPoints]
#
#
#
#
#
# def get_entourage(sgmf, circle):
#     N=47
#     n=int((N-1)/2)
#     srcPoints=np.zeros((N**2, 2))
#     dstPoints=np.zeros((N**2, 2))
#     k=0
#     for i in range(-n,n):
#         for j in range(-n,n):
#             camPix=np.array([circle[0],circle[1],1])+np.array([i,j,0]) #coordonées homogènes
#             projPix=sgmf.get_value(camPix) #coordonées homogènes
#             srcPoints[k,:]=camPix[:2]
#             dstPoints[k,:]=projPix[:2]
#             k+=1
#     return srcPoints, dstPoints
