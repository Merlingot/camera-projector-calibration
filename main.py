""" Fichier pour détecter les centres des cercles et écrire les coordonées dans un fichier .txt"""
import os
import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt

# Personal imports :
import sgmf
import tsai
from detectCenters import detect_centers
from circlesgrid import getAsymCirclesObjPoints
import unwrapping.util as util

# Paramètres =============================================================
# Résolution du projecteur en pixel
projSize=(1920,1200)
# Data:
SERIE="13_11_2020/serie_gp_1"
# Damier
points_per_row=8; points_per_colum=8
circleDiameter=10e-2
circleSpacing=20e-2
paperMargin=20e-2 # À trouver
# ========================================================================

# Paths:
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
patternSize = (points_per_row, points_per_colum)
realPatternSize=(2*points_per_row, points_per_colum)
# Points 3d
offset=paperMargin+(circleSpacing+circleDiameter)/2
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
# ax.set_aspect('equal')
# ax.plot(objp[:,0], objp[:,2],'o')
# plt.savefig('{}damier_vue_haut.png'.format(verifPath), format="png")



# Centre des cercles dans les deux images (cam et projecteur) -----------------
# 1. Points dans le référentiel monde
objectPoints=[objpR.astype(np.float32), objpL.astype(np.float32)]

# 2. Détection des centres dans l'image de la caméra
imagePoints, imgp = detect_centers(patternSize, color, gray, verifPath)

# 3. Extraction des coordonnées des points dans le plan image du projecteur
def get_projPoints(sgmf, imagePoints):
    projPoints = imagePoints.copy()
    for i in range(imagePoints.shape[0]):
        circle=imagePoints[i][0] #Format bizare
        # sans transformation affine:
        p=sgmf.get_value(circle)
        projPoints[i,0,:]=np.array([ np.array([p[0],p[1]]).astype(np.float32) ]) #Format bizare
    return projPoints
SGMF=sgmf.sgmf(sgmfPath, projSize, shadowMaskName=None)
projp=get_projPoints(SGMF, imgp)
# Vérification
# plt.figure()
# plt.title('Vérification points images')
# plt.imshow(SGMF.channelX)
# plt.plot(imagePoints[0][:,0,0],imagePoints[0][:,0,1], 'o')
# plt.plot(imagePoints[1][:,0,0],imagePoints[1][:,0,1], 'o')
# plt.savefig('{}superposionSGMF.png'.format(verifPath))

# 4. Écriture dans un fichier :
file = open("{}points.txt".format(pointsPath),"w")
for i in range(imgp.shape[0]):
    point2dCam=imgp[i][0] #array(array(u,v))
    point2dProj=projp[i][0]#array(array(u,v))
    point3d=objp[i]
    line=[ "{} ".format(point3d[0]), "{} ".format(point3d[1]), "{} ".format(point3d[2]), "{} ".format(point2dCam[0]), "{} ".format(point2dCam[1]), "{} ".format(point2dProj[0]), "{} \n".format(point2dProj[1]) ]
    file.writelines(line)
file.close()



























# # Calibration de la caméra (estimation)  ----------------------------------
# # Format de imgPoints et objPoints (pour la calibration) À CORRIGER
# objPoints=[objpR.astype(np.float32)]
# imgPoints=[imagePoints[0]]
# # calibrateCamera
# ret, cameraMatrix, camDistCoeffs, _, _ = cv.calibrateCamera(objPoints, imgPoints, gray.shape[::-1],None,None)
#
# #Tsai
# dataPath="data/13_11_2020/serie_gp_1"
# imageShape=gray.shape
# pointsPath=os.path.join(dataPath,"points/points.txt")
# campixelSize=(3.45e-6, 3.45e-6)
# R, T, f = tsai.main(pointsPath, imageShape, campixelSize)
#
#
# def draw(img, origin, imgpts):
#     # BGR
#     img = cv.line(img, tuple(origin[0].ravel()), tuple(imgpts[0].ravel()), (255,0,0), 5) #X
#     img = cv.line(img, tuple(origin[0].ravel()), tuple(imgpts[1].ravel()), (0,255,0), 5)
#     img = cv.line(img, tuple(origin[0].ravel()), tuple(imgpts[2].ravel()), (0,255,255), 5) # Z en jaune
#     return img
#
# # Vérification de la calibration de la caméra en reprojection:
# # Montrer axes
# axis = np.float32([[1,0,0], [0,1,0], [0,0,1]]).reshape(-1,3)*1
# ret, rvecs, tvecs = cv.solvePnP(objPoints[0], imgPoints[0], cameraMatrix, camDistCoeffs)
# axisProj, jac = cv.projectPoints(axis, rvecs, tvecs, cameraMatrix, camDistCoeffs)
# origin = np.float32([[0,0,0]]).reshape(-1,1)
# originProj , jac = cv.projectPoints(origin, rvecs, tvecs, cameraMatrix, camDistCoeffs)
# img = draw(color.copy(), originProj[0], axisProj)
# cv.imwrite('{}reprojection_axes.png'.format(verifPath), img)
# # À droite
# pts = objpR.astype(np.float32)
# ptsProj, jac = cv.projectPoints(pts, rvecs, tvecs, cameraMatrix, camDistCoeffs)
# img = cv.drawChessboardCorners(color.copy(), patternSize, ptsProj, 1)
# cv.imwrite('{}reprojection_droite.png'.format(verifPath), img)
# # À gauche
# pts = objpL.astype(np.float32)
# ptsProj, jac = cv.projectPoints(pts, rvecs, tvecs, cameraMatrix, camDistCoeffs)
# img = cv.drawChessboardCorners(color.copy(), patternSize, ptsProj, 1)
# cv.imwrite('{}reprojection_gauche.png'.format(verifPath), img)
#
#
#
# # Calibration du projecteur
# ret, projMatrix, projDistCoeffs, prvecs, ptvecs = cv.calibrateCamera(objPoints, projPoints, gray.shape[::-1],None,None)
#
# # Calibration stéréo  ----------------------------------
# projectorPoints=[ projPoints[0], get_projPoints(SGMF, imagePoints[1])[0] ]
# plt.figure()
# plt.title('Points plan image projecteur trouvés avec sgmf')
# plt.plot(projectorPoints[0][:,0,0], projectorPoints[0][:,0,1], 'o')
# plt.plot(projectorPoints[1][:,0,0], projectorPoints[1][:,0,1], 'o')
# plt.savefig('{}points_projecteur.png'.format(verifPath))
#
# retval, cameraMatrix, camDistCoeffs, projMatrix, projDistCoeffs, R, T, E, F, perViewErrors = cv.stereoCalibrateExtended(objectPoints, imagePoints, projectorPoints, cameraMatrix, camDistCoeffs, projMatrix, projDistCoeffs, imageSize, np.zeros((3,3)), np.zeros(3))









































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
