import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
import sgmf

# Nombre de cercles
NB = 2*(4*6+4*5)
points_per_row=4;points_per_colum=11
patternSize = (points_per_row, points_per_colum)
# Largeur d'un cercle
circleDiameter = 1.5e-2
circleSpacing = 2e-2
# Resolution du projecteur
projSize=(800,600)
# Images
imgPath="data/serie_1/nofringe/noFringe.png"
sgmfPath="data/serie_1/cam_match.png"
#objectPoints:

# Liste avec le nom de chaque image
fname = glob.glob(imgPath)
# sgmfname = glob.glob(sgmfPath)
# lire les images
color=cv.imread(fname[0])
gray = cv.cvtColor(color , cv.COLOR_BGR2GRAY)
# sgmf = sgmf.sgmf(sgmfname[0], projSize, shadowMaskName=None)

# ---------------------------

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# finder
finderParams=cv.CirclesGridFinderParameters()
# Setup SimpleBlobDetector parameters.
blobParams = cv.SimpleBlobDetector_Params()
# Change thresholds
blobParams.minThreshold = 8
blobParams.maxThreshold = 255
# Filter by Area.
blobParams.filterByArea = True
blobParams.minArea = 100    # minArea may be adjusted to suit for your experiment
blobParams.maxArea = 750   # maxArea may be adjusted to suit for your experiment
# Filter by Circularity
blobParams.filterByCircularity = True
blobParams.minCircularity = 0.1
# Filter by Convexity
blobParams.filterByConvexity = True
blobParams.minConvexity = 0.87
# Filter by Inertia
blobParams.filterByInertia = True
blobParams.minInertiaRatio = 0.01
# Create a detector with the parameters
blobDetector = cv.SimpleBlobDetector_create(blobParams)
keypoints = blobDetector.detect(gray) # Detect blobs.

im_with_keypoints = cv.drawKeypoints(gray, keypoints, np.array([]), (255,0,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
im_with_keypoints_gray = cv.cvtColor(im_with_keypoints, cv.COLOR_BGR2GRAY)
ret, centers = cv.findCirclesGrid(im_with_keypoints_gray, patternSize, cv.CALIB_CB_ASYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING, blobDetector, finderParams)


objp=np.zeros((NB,3))
objPoints=[]; imgPoints=[]

if ret == True:
    objPoints.append(objp)
    centers2 = cv.cornerSubPix(gray, centers, patternSize, (-1,-1), criteria)
    imgPoints.append(centers2)

    # Draw and display the corners.
    img = cv.drawChessboardCorners(color, patternSize, centers2, ret)
    cv.imwrite("centers.png",img)

    file = open("points.txt","w")
    for i in range(centers2.shape[0]):
        point2d=centers2[i][0]
        point3d=objp[i]
        line=[ "{} ".format(point3d[0]), "{} ".format(point3d[1]), "{} ".format(point3d[2]), "{} ".format(point2d[0]), "{} \n".format(point2d[1]) ]
        file.writelines(line)
    file.close()









#
