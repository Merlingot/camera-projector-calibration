""" Fichier pour...
-> La camera: détecter les centres des cercles et écrire les coordonées dans un fichier .txt
-> Le projecteur: utiliser la sgmf pour trouver le centre des cercles et etc."""

import os
import numpy as np
import cv2 as cv
import glob

import sgmf
from detect import detect_centers, detect_corners
from grid import getAsymCirclesObjPoints, getCorners
from util import outputClean


def get_projPoints(sgmf,imgp):
    projp=imgp.copy()
    for i in range(projp.shape[0]):
        circle=imgp[i][0] #Format bizare
        # transformation affine
        srcPoints, dstPoints = get_47(sgmf,circle)
        mat,_ = cv.estimateAffine2D(srcPoints, dstPoints)
        A = np.array([[mat[0,0],mat[0,1]],[mat[1,0],mat[1,1]]]); b=mat[:,2]
        p=A@circle + b
        projp[i,0,:] = np.array([p.astype(np.float32)])
    return projp


def get_47(sgmf, circle):
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



def get_objp(points_per_row, points_per_colum, paperMargin, spacing, circleDiameter, option, motif, damier=None):

    patternSize=(points_per_row, points_per_colum)

    if motif=='cercle':
        if option=='tsai':

            offset=paperMargin+(spacing+circleDiameter)/2

            objpR = getAsymCirclesObjPoints(points_per_colum, points_per_row, circleDiameter+spacing, offset, 0, "xy")
            objpL = getAsymCirclesObjPoints(points_per_colum, points_per_row, circleDiameter+spacing, offset, 0, "yz")
            objp=np.concatenate((objpR, objpL))
            return patternSize, objp.astype(np.float32)

        elif option=='zhang':

            offset=(spacing+circleDiameter)/2

            objpR = getAsymCirclesObjPoints(points_per_colum, points_per_row, circleDiameter+spacing, offset, 0, "xy")
            objp=np.concatenate((objpR, objpR))
            return patternSize, objp.astype(np.float32)
        else:
            print("Options pour la calibration : 'tsai' ou 'zhang' ")
            return 0,0


    elif motif=='square':
        offset = paperMargin+spacing

        if damier=='simple':
            objp = getCorners(points_per_colum, points_per_row, spacing, offset, 0, "xy")
            return patternSize, objp.astype(np.float32)

        else:
            if option == 'tsai':
                objpR = getCorners(points_per_colum, points_per_row, spacing, offset, 0, "xy")
                objpL = getCorners(points_per_colum, points_per_row, spacing, offset, 0, "yz")
                objp=np.concatenate((objpR, objpL))
                return patternSize, objp.astype(np.float32)
            elif option=='zhang':
                objpR = getCorners(points_per_colum, points_per_row, spacing, offset, 0, "xy")
                objp=np.concatenate((objpR, objpR))
                return patternSize, objp.astype(np.float32)
            else:
                print("Options pour la calibration : 'tsai' ou 'zhang' ")
                return 0,0

    else:
        print("Options pour le motif : 'cercle' ou 'square' ")
        return 0,0





def camera_centers(points_per_row, points_per_colum, paperMargin, spacing, circleDiameter, noFringePath, verifPath, option, motif, damier='double', pointsPath=None):

    # Clean paths:
    output_paths=[verifPath, pointsPath]
    outputClean(output_paths)

    # Lire l'image
    color=cv.imread(noFringePath)
    gray=cv.cvtColor(color , cv.COLOR_BGR2GRAY)

    # Points 3d
    patternSize, objp = get_objp(points_per_row, points_per_colum, paperMargin, spacing, circleDiameter, option, motif, damier)

    if motif=='cercle':
    # # Détection des centres dans l'image de la caméra (motif cercles)
        _, imgp = detect_centers(patternSize, color, gray, verifPath)
    elif motif=='square':
        # Détection des coins dans l'image de la caméra (motif carrés)
        _, imgp = detect_corners(patternSize, color, gray, damier, verifPath)
    else:
        print("motif: 'square' ou 'cercle' ")

    # 4. Écriture dans un fichier :
    fileCam = open("{}points_camera.txt".format(pointsPath),"w")
    for i in range(imgp.shape[0]):
        point2dCam=imgp[i][0] #array(array(u,v))
        point3d=objp[i]
        line=[ "{} ".format(point3d[0]), "{} ".format(point3d[1]), "{} ".format(point3d[2]), "{} ".format(point2dCam[0]), "{} \n ".format(point2dCam[1]) ]
        fileCam.writelines(line)
    fileCam.close()
    return objp.astype(np.float32), imgp.astype(np.float32)


def proj_centers(objp, imgp, projSize, sgmfPath, pointsPath=None):

    # Extraction des coordonnées des points dans le plan image du projecteur
    SGMF=sgmf.sgmf(sgmfPath, projSize, shadowMaskName=None)
    projp=get_projPoints(SGMF, imgp)

    # # 4. Écriture dans un fichier :
    # fileProj = open("{}points_proj.txt".format(pointsPath),"w")
    # for i in range(imgp.shape[0]):
    #     point2dProj=projp[i][0]#array(array(u,v))
    #     point3d=objp[i]
    #     line=[ "{} ".format(point3d[0]), "{} ".format(point3d[1]), "{} ".format(point3d[2]), "{} ".format(point2dProj[0]), "{} \n".format(point2dProj[1]) ]
    #     fileProj.writelines(line)
    # fileProj.close()
    return projp
