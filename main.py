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


def get_projPoints(sgmf,imgp):
    projp=imgp.copy()
    for i in range(projp.shape[0]):
        circle=imgp[i][0] #Format bizare
        # transformation affine
        srcPoints, dstPoints = get_entourage(sgmf,circle)
        mat,_ = cv.estimateAffine2D(srcPoints, dstPoints)
        A = np.array([[mat[0,0],mat[0,1]],[mat[1,0],mat[1,1]]]); b=mat[:,2]
        p=A@circle + b
        projp[i,0,:] = np.array([p.astype(np.float32)])
        # sans transformation affine:
        #         p=sgmf.get_value(circle)
        #         projPoints[i,0,:]=np.array([ np.array([p[0],p[1]]).astype(np.float32)
    return projp


def get_entourage(sgmf, circle):
    N=47
    n=int((N-1)/2)
    srcPoints=np.zeros((N**2, 2))
    dstPoints=np.zeros((N**2, 2))
    k=0
    for i in range(-n,n):
        for j in range(-n,n):
            camPix=np.array([circle[0],circle[1],1])+np.array([i,j,0]) #coordonées homogènes
            projPix=sgmf.get_value(camPix) #coordonées homogènes
            srcPoints[k,:]=camPix[:2]
            dstPoints[k,:]=projPix[:2]
            k+=1
    return srcPoints, dstPoints


def main(points_per_row, points_per_colum, noFringePath, sgmfPath, verifPath, pointsPath ):

    # Clean paths:
    output_paths=[verifPath, pointsPath]
    util.outputClean(output_paths)

    # Lire l'image
    color=cv.imread(noFringePath)
    gray = cv.cvtColor(color , cv.COLOR_BGR2GRAY)
    # Résolution caméra:
    imageSize=gray.shape

    # Damier
    patternSize = (points_per_row, points_per_colum)
    offset=paperMargin+(circleSpacing+circleDiameter)/2
    objpR = getAsymCirclesObjPoints(points_per_colum, points_per_row, circleDiameter+circleSpacing, offset, 0, "xy")
    objpL = getAsymCirclesObjPoints(points_per_colum, points_per_row, circleDiameter+circleSpacing, offset, 0, "yz")
    objp=np.concatenate((objpR, objpL))


    # Centre des cercles:
    # 1. Points dans le référentiel monde
    # objectPoints=[objpR.astype(np.float32), objpL.astype(np.float32)]

    # 2. Détection des centres dans l'image de la caméra
    _, imgp = detect_centers(patternSize, color, gray, verifPath)

    # 3. Extraction des coordonnées des points dans le plan image du projecteur
    SGMF=sgmf.sgmf(sgmfPath, projSize, shadowMaskName=None)
    projp=get_projPoints(SGMF, imgp)


    # 4. Écriture dans un fichier :
    file = open("{}points.txt".format(pointsPath),"w")
    for i in range(imgp.shape[0]):
        point2dCam=imgp[i][0] #array(array(u,v))
        point2dProj=projp[i][0]#array(array(u,v))
        point3d=objp[i]
        line=[ "{} ".format(point3d[0]), "{} ".format(point3d[1]), "{} ".format(point3d[2]), "{} ".format(point2dCam[0]), "{} ".format(point2dCam[1]), "{} ".format(point2dProj[0]), "{} \n".format(point2dProj[1]) ]
        file.writelines(line)
    file.close()
    return objp, imgp, projp

objp, imgp, projp = main(points_per_row, points_per_colum, noFringePath, sgmfPath, verifPath, pointsPath)
