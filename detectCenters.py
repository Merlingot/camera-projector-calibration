import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def detect_centers(patternSize, objp, color, gray, verifPath, pointsPath):
    # Nombre de cercles
    points_per_row=patternSize[0]; points_per_colum=patternSize[1]
    NB=points_per_row*points_per_colum*2

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Finder parameters
    finderParams=cv.CirclesGridFinderParameters()

    # SimpleBlobDetector parameters
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

    # Detect centers
    keypoints = blobDetector.detect(gray) # Detect blobs.
    im_with_keypoints = cv.drawKeypoints(gray, keypoints, np.array([]), (255,0,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_with_keypoints_gray = cv.cvtColor(im_with_keypoints, cv.COLOR_BGR2GRAY)

    x = []
    for i in range(len(keypoints)):
        x.append(keypoints[i].pt[0])
    mean = np.mean(np.array(x))

    # Mask :
    xIndex = np.zeros((gray.shape[0],1))
    for i in range(1,gray.shape[1]):
        row = np.ones((gray.shape[0],1))*i
        xIndex = np.concatenate((xIndex, row ), axis=1)
    maskR=xIndex>mean
    maskL=xIndex<mean
    grayR=gray*maskR
    grayL=gray*maskL

    # Find circles
    retR, centersR = cv.findCirclesGrid(grayR, patternSize, cv.CALIB_CB_ASYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING, blobDetector, finderParams)
    retL, centersL = cv.findCirclesGrid(grayL, patternSize, cv.CALIB_CB_ASYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING, blobDetector, finderParams)

    ret=retR*retL
    if ret == True:
        imgpR = cv.cornerSubPix(gray, centersR, patternSize, (-1,-1), criteria)
        imgpL = cv.cornerSubPix(gray, centersL, patternSize, (-1,-1), criteria)

        imgp=np.concatenate((imgpR, imgpL))
        # Draw and display the corners.
        imgR = cv.drawChessboardCorners(color.copy(), patternSize, imgpR, ret)
        cv.imwrite("{}centersR.png".format(verifPath),imgR)
        imgL = cv.drawChessboardCorners(color.copy(), patternSize, imgpL, ret)
        cv.imwrite("{}centersL.png".format(verifPath),imgL)
        # write to file
        file = open("{}points.txt".format(pointsPath),"w")
        for i in range(imgp.shape[0]):
            point2d=imgp[i][0] #array(array(u,v))
            point3d=objp[i]
            line=[ "{} ".format(point3d[0]), "{} ".format(point3d[1]), "{} ".format(point3d[2]), "{} ".format(point2d[0]), "{} \n".format(point2d[1]) ]
            file.writelines(line)
        file.close()
        return imgp
    else:
        print("Fail")
