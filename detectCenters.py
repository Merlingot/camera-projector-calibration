""" Fonction pour détecter le centre des cercles sur un tsai grid """

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def detect_centers(patternSize, color, gray, verifPath, pointsPath=None):
    # Nombre de cercles
    points_per_row=patternSize[0]; points_per_colum=patternSize[1]
    NB=points_per_row*points_per_colum*2

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Finder parameters
    finderParams=cv.CirclesGridFinderParameters()

    # SimpleBlobDetector parameters
    blobParams = cv.SimpleBlobDetector_Params()
    # Thresholds
    blobParams.minThreshold = 8
    blobParams.maxThreshold = 255
    # # Filter by Area.
    blobParams.filterByArea = True
    blobParams.minArea = 200    # minArea may be adjusted to suit experiment
    blobParams.maxArea = 10000
    # Filter by Circularity
    blobParams.filterByCircularity = True
    blobParams.minCircularity = 0.8
    blobParams.maxCircularity = np.inf
    # Filter by Convexity
    blobParams.filterByConvexity = True
    blobParams.minConvexity = 0.9
    blobParams.maxConvexity = np.inf
    # Filter by Inertia
    blobParams.filterByInertia = True
    blobParams.minInertiaRatio = 0.01
    blobParams.maxInertiaRatio = np.inf
    # Create a detector with the parameters
    blobDetector = cv.SimpleBlobDetector_create(blobParams)

    # Detect centers
    keypoints = blobDetector.detect(gray) # Detect blobs.
    im_with_keypoints = cv.drawKeypoints(gray.copy(), keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_with_keypoints_gray = cv.cvtColor(im_with_keypoints, cv.COLOR_BGR2GRAY)
    cv.imwrite("{}blobDetected.png".format(verifPath),im_with_keypoints)


    x = []
    for i in range(len(keypoints)):
        x.append(keypoints[i].pt[0])
    mean = np.mean(np.array(x))

    # Mask :
    xIndex = np.indices(gray.shape)[1]

    maskR=xIndex>mean*0.975
    maskL=xIndex<mean*1.025
    grayR=gray*maskR
    grayL=gray*maskL

    # Detect centers
    # right
    blobDetectorR = cv.SimpleBlobDetector_create(blobParams)
    keypointsR = blobDetectorR.detect(grayR) # Detect blobs.
    keypoints_colorR = cv.drawKeypoints(grayR.copy(), keypointsR, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    keypoints_grayR = cv.cvtColor(keypoints_colorR, cv.COLOR_BGR2GRAY)
    cv.imwrite("{}blobDetectedR.png".format(verifPath),keypoints_grayR)
    #left
    blobDetectorL = cv.SimpleBlobDetector_create(blobParams)
    keypointsL = blobDetector.detect(grayL) # Detect blobs.
    keypoints_colorL = cv.drawKeypoints(grayL.copy(), keypointsL, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    keypoints_grayL = cv.cvtColor(keypoints_colorL, cv.COLOR_BGR2GRAY)
    cv.imwrite("{}blobDetectedL.png".format(verifPath),keypoints_grayL)

    # Find circles
    retR, centersR = cv.findCirclesGrid(keypoints_grayR, patternSize,cv.CALIB_CB_SYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING, blobDetectorR, finderParams)
    retL, centersL = cv.findCirclesGrid(keypoints_grayL, patternSize, cv.CALIB_CB_SYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING, blobDetectorL, finderParams)



    imgPoints=[]
    ret=retR*retL
    if ret == True:
        imgpR = cv.cornerSubPix(gray, centersR, patternSize, (-1,-1), criteria)
        imgpL = cv.cornerSubPix(gray, centersL, patternSize, (-1,-1), criteria)

        imgPoints.append(imgpR)
        imgPoints.append(imgpL)
        imgp=np.concatenate((imgpR, imgpL)) #Meme format que objp

        # Draw and display the corners.
        imgR = cv.drawChessboardCorners(color.copy(), patternSize, imgpR, ret)
        cv.imwrite("{}centersR.png".format(verifPath),imgR)
        imgL = cv.drawChessboardCorners(color.copy(), patternSize, imgpL, ret)
        cv.imwrite("{}centersL.png".format(verifPath),imgL)



        return imgPoints, imgp
    else:
        print("Fail")
