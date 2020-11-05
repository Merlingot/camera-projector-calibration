
import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
import sgmf
# from tsai import calibrate
# from blob import detect_centers
from circlesgrid import getAsymCirclesObjPoints

def T(d,a):
    return np.array([a,0,a+d])

def Ry(o):
    R = np.zeros((3,3))
    R[0,0]=np.cos(o)
    R[0,2]=np.sin(o)
    R[1,1]=1
    R[2,0]=-np.sin(o)
    R[2,2]=np.cos(o)
    return R

# Damier
points_per_row=4; points_per_colum=11
patternSize = (points_per_row, points_per_colum)
circleDiameter=1.5e-2
circleSpacing=2e-2
d = (points_per_row-1)
# Points 3d
objp1 = getAsymCirclesObjPoints(points_per_colum, points_per_row, 1, 0, 0, "xy")
objp2 = getAsymCirclesObjPoints(points_per_colum, points_per_row, 1, d, 0, "yz")
objp=np.concatenate((objp1, objp2))
#Résolution du projecteur en pixel
projSize=(800,600)

detect_centers(patternSize, objp, color, gray)


#
# def main(imgPATH, sgmfPATH, patternSize, projSize, objp):
"""
Args:
    imgPath :       path to camera image directory
    sgmfPath :      path to sgmf directory
    patternSize :   tuple
                    (number of circles in the left/right direction, number of circles in the up/down direction)
    projSize:       tuple (width, height)
                    Projector's resolution in pixel
    objp :          array(array(x,y,z))
                    Coordonnées des centres des cercles sur la grille
"""

# Nom des images dans une liste
fnames = glob.glob(imgPath)
sgmfnames = glob.glob(sgmfPath)

if (len(fnames)>0 and len(sgmfnames)>0) :
    # lire les images
    color=cv.imread(fnames[0])
    gray = cv.cvtColor(color , cv.COLOR_BGR2GRAY)
    # sgmf = sgmf.sgmf(sgmfnames[0], projSize, shadowMaskName=None)

detect_centers(patternSize, objp, color, gray)

    # 1. Calibration intrinsèque de la caméra:

    objectPoints, imagePoints = detect_centers()

    # 2. Extraction des coordonées des points dans le plan image du projecteur
    projPoints=get_projPoints(sgmf, imagePoints, objp.copy())

    # 3. Calibration intrinsèque du projecteur
    projMatrix, projDistCoeffs = proj_intrinsic(objectPoints, projPoints, projSize)

    # 4. Calibration stereo
    imageSize=gray.shape
    retval, cameraMatrix, camDistCoeffs, projMatrix, projDistCoeffs, R, T, E, F, perViewErrors, flags = cv.stereoCalibrateExtended(objectPoints, imagePoints, projoints, cameraMatrix, camDistCoeffs, projMatrix, projDistCoeffs, imageSize)





def cam_intrinsic(patternSize, objectPoints, gray, criteria=None):
    """
    Find the intrinsic calibration parameters (camera matrix and distortion coefficients) of one camera

    Args:
        patternSize :   tuple(points_per_row, points_per_colum)
                        Number of circles per row and column
        objectPoints :  array ( array(x,y,z) )
                        Real world coordinates of the circle's centers corners.
        gray :          image
        criteria :      termination criteria
    Returns:
        objectPoints :  array [ array(x,y,0) ] ]
                        Real world coordinates of the circle's centers corners.
        imagePoints :   list[ array [ array(u,v,0) ] ]
                        Pixel coordinates of  the circle's centers for each frame in the camera coord.
        cameraMatrix : camera matrix (K)
        distCoeffs : distortion coefficient
    """
    # Trouver le centre des cercles
    retval, imagePoints	= cv.findCirclesGrid(gray, patternSize)

    # Calibration de la caméra --------------------
    ret, cameraMatrix, distCoeffs, _ , _ = cv.calibrateCamera([objectPoints], [imagePoints], gray.shape[::-1], None, None)

    return  objectPoints, imagePoints, cameraMatrix, distCoeffs



def get_projPoints(sgmf, imagePoints, projPoints):
    """ Find the corresponding pixel coordinates in the projector plane of the imagePoints.

    Args:
        sgmf :
        imagePoints : array(N,(u,v,1))
        projPoints: array(N,(u,v,1))
    """

    for i in range(imgPoints.shape[0]):
        circle=imgPoints[i]
        scrPoints, dstPoints = get_entourage(sgmf,circle)
        matH, _ = cv.findHomography(srcPoints, dstPoints)
        projPoints[i,:] = matH@circle

    return projPoints


def get_entourage(sgmf, circle):
    N=47
    n=(N-1)/2
    srcPoints=np.zeros(N**2, 3)
    dstPoints=np.zeros(N**2, 3)
    k=0
    for i in range(-n,n):
        for j in range(-n,n):
            camPix=circle+np.array([i,j,0])
            projPix=sgmf.get_value(camPix)
            srcPoints[k,:]=camPix
            dstPoints[k,:]=projPix
            k+=1
    return srcPoints, dstPoints



def proj_intrinsic(objectPoints, projPoints, projSize):
    """
    Find the intrinsic calibration parameters (projector matrix and distortion coefficient) of one projector.
    Args:
        objectPoints :  array(N,(x,y,z))
                        For one frame, coordinates of circles, in world coordinates.
        projPoints :    array(N,(u,v,1))
                        For one frame, pixel coordinates of circles in the projector coordinates.
        projSize :      tuple(x,y)
                        Projector's resolution.

    Returns:
        projMatrix:     Projector instrinsic matrix (K)
        projDistCoeffs: Projector distortion coefficient
    """

    ret, projMatrix, projDistCoeffs, _, _ = cv.calibrateCamera([objectPoints], [projPoints], projSize, None, None)

    return  projMatrix, projDistCoeffs
