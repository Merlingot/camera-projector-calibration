""" Calibration functions. See openCV documentation for ore information"""
import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt

# Nombre de coins internes du damier
NB_CORNER_WIDTH=5
NB_CORNER_HEIGHT=4
# Largeur d'un carré du damier en mètres (damier réel)
squareSize = 1
FLIP=0 # flip par rapport à y
projDIMENSION=np.array([800,600])


def main(imgPATH, sgmfPATH, NB_CORNER_WIDTH, NB_CORNER_HEIGHT, squareSize, projDIMENSION):
    """
    Finds and write corners coordinates for each images in PATH directory. Writes all information necessary to run Takahashi's algorithm after.
    Args:
        imgPATH : path to images directory
        sgmfPATH : path to sgmf directory
        NB_CORNER_WIDTH : number of internal corners in the left/right direction
        NB_CORNER_HEIGHT : number of internal corners in the up/down direction
        squareSize : size of chessboard square side in meter
    """

    CHECKERBOARD = ( NB_CORNER_WIDTH, NB_CORNER_HEIGHT)
    # Prepare object points (corner coordinates)
    objp = np.zeros((NB_CORNER_WIDTH*NB_CORNER_HEIGHT,3), np.float32)
    # objp[:,:2] = np.mgrid[0:NB_CORNER_WIDTH,0:NB_CORNER_HEIGHT].T.reshape(-1,2)
    k=0
    for j in range(NB_CORNER_HEIGHT):
        for i in range(NB_CORNER_WIDTH):
            objp[k] = [i,NB_CORNER_HEIGHT-j-1,0]
            k+=1
    objp = objp*squareSize

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Images to analyse
    fnames = glob.glob(imgPATH)
    sgmfnames = glob.glob(sgmfPATH)


    if (len(fnames)>0 and len(sgmfnames)>0) :

        # 1. Calibration intrinsèque de la caméra:

        objectPoints, imagePoints, cameraMatrix, distCoeffs = cam_intrinsic(NB_CORNER_WIDTH, NB_CORNER_HEIGHT, CHECKERBOARD, objp, criteria, fnames)

        # Undistort ?

        # 2. Extraction des coordonées des points dans le plan image du projecteur
        projCorners=np.zeros(objp.shape())
        projPoints=get_projPoints(sgmfnames, imagePoints, projDIMENSION, projCorners)

        # 3. Calibration intrinsèque du projecteur
        projMatrix, distCoeffs = proj_intrinsic(objectPoints, projPoints, imageSize)

        # 4. Calibration stereo
        retval, cameraMatrix, camDistCoeffs, projMatrix, projDistCoeffs, R, T, E, F, perViewErrors, flags = cv.stereoCalibrateExtended( objectPoints, imagePoints, projectorPoints, cameraMatrix, camDistCoeffs, projMatrix, projDistCoeffs, imageSize)

        return R, T
    else :
        print('No images found')




def cam_intrinsic(NB_CORNER_WIDTH, NB_CORNER_HEIGHT, CHECKERBOARD, objp, criteria, fnames):
    """
    Find the intrinsic calibration parameters (camera matrix and distortion coefficient) of one camera

    Args:
        NB_CORNER_WIDTH : number of internal corners in the left/right direction
        NB_CORNER_HEIGHT : number of internal corners in the up/down direction
        objp : Coordinates of corners in chessboard plane
        criteria : termination criteria
        fnames : path to images to analyses (Serie of cherckerboard frame)

    Returns:
        objectPoints : {[[x,y,0]]} real world coordinates of checkerboard corners for each frame in checkerboard coord.
        imagePoints : {[[u,v,1]]} pixel coordinates of checkerboard corners for each frame in the camera coord.
        cameraMatrix : camera matrix (K)
        distCoeffs : distortion coefficient
    """
    objectPoints = [] # 3d point in real world space (Coordinates of Cherckerboard corners in the "checkerboard plane)
    imagePoints = [] # 2d points in image plane. (Coordinates of Cherckerboard corners for each frame)

    # Pour chaque image de damier:
    for fname in fnames:
        img = cv.imread(fname);
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objectPoints.append(objp)
            cornersSubPix = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imagePoints.append(cornersSubPix)
            # Draw and save the corners ----------------------------
            cv.drawChessboardCorners(img, CHECKERBOARD, cornersSubPix, ret)
            cv.imwrite('corners_{}.png'.format(fname), cv.flip(img, FLIP))

    cv.destroyAllWindows()
    # Calibration de la caméra --------------------
    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera(objectPoints, imagePoints, gray.shape[::-1], None, None)

    return  objectPoints, imgagePoints, cameraMatrix, distCoeffs



def get_projPoints(sgmfnames, imagePoints, projDIMENSION, projCorners):
    """ Find the corresponding pixel coordinates in the projector plane of the imagePoints.

    Args:
        sgmfnames : [str] list of the sgmf file names
        imagePoints : [ [(u,v,1)] ] corner coords in image plane for each frame
        projDimension: array([wx, wy]) dimension du projecteur en pixel
        projCorners = np.zeros((NB_CORNER_WIDTH*NB_CORNER_HEIGHT,3))
    """
    projPoints=[]

    wx, wy = projDimension[0], projDimension[1]

    # For each frame:
    for sgmfname, imgCorners in zip(sgmfnames,imagePoints):

        sgmfXY = cv2.imread(sgmfname,-1)/65535
        sgmf = np.zeros( [sgmfXY.shape[0], sgmfXY.shape[1],2] ) #SHAPE Lignes,Colonnes,CHANNEL
        sgmf[:,:,0] = sgmfXY[:,:,2] * wx # channel X
        sgmf[:,:,1] = sgmfXY[:,:,1] * wy # channel Y

        # For each corner in each frame
        for imgCorner,projCorner in zip(imgCorners,projCorners): #?
            scrPoints, dstPoints = get_entourage(sgmf, imgCorner)
            matH, mask = cv.findHomography(srcPoints, dstPoints)
            projCorner = matH@imgCorner
            projPoints.append(projCorner)

    return projPoints


def get_entourage(sgmf, corner):
    N=47
    n=(N-1)/2
    srcPoints=np.zeros(N**2, 3)
    dstPoints=np.zeros(N**2, 3)
    k=0
    for i in range(-n,n):
        for j in range(-n,n):
            camPix=corner+np.array([i,j,0])
            projPix=SGMF(sgmf,camPix)
            srcPoints[k,:]=camPix
            srcPoints[k,:]=projPix
            k+=1
    return srcPoints, dstPoints

def SGMF(sgmf, vecPix):
    u,v = int(np.round(vecPix[0])), int(np.round(vecPix[1]))
    # INDEXATION LIGNE (v), COLONNE (u) !!!!!!
    ex, ey = self.sgmf[v,u,0], self.sgmf[v,u,1] #les channels
    return np.array([ex,ey,1])


def proj_intrinsic(objectPoints, projPoints, imageSize):
    """
    Find the intrinsic calibration parameters (projector matrix and distortion coefficient) of one projector given projPoints
    Args:
        objectPoints : {[[x,y,z]]} coordinates of chessboard corners, for each chessboard, in world coordinates, taken from the camera calibration
        projPoints : {[[u,v]]} pixel coordinates of checkerboard corners for each frame in the projector coord.
        imageSize : ?
    Returns:
        objectPoints : {[[x,y,0]]} real world coordinates of checkerboard corners for each frame in checkerboard coord.
        imgaMatrix : projector instrinsic matrix (K)
        distCoeffs : distortion coefficient
    """

    ret, projMatrix, distCoeffs, _, _ = cv.calibrateCamera(objectPoints, projPoints,  imageSize, None, None)

    return  projMatrix, distCoeffs





def undistort(mtx, dist, img):
    """ Undistort one image """
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

def reprojection_err(objp, corners2, mtx, dist, rvecs, tvecs):
    """ reprojection_err for one image """
    imgpoints2, _ = cv.projectPoints(objp, rvecs, tvecs, mtx, dist)
    error = cv.norm(corners2, imgpoints2, cv.NORM_L2)/len(imgpoints2)
    return error

def find_corners(NB_CORNER_WIDTH, NB_CORNER_HEIGHT, CHECKERBOARD, criteria, objp, dst, mtx, dist):
    """
    Find corners coordinates in one (undistorted) image
    Args:
        NB_CORNER_WIDTH : number of internal corners in the left/right direction
        NB_CORNER_HEIGHT : number of internal corners in the up/down direction
        squareSize : size of chessboard square in meter
        dst : undistorted image
    Returns :
        ret (Bool) : succes of findChessboardCorners
        imgpoints : list of chessboardcorners in image coordinates
        objpoints : list of chessboardcorners in world coordinates
    """
    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)
    # If found:
    if ret == True:
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
        err = reprojection_err(objp, corners2, mtx, dist, rvecs, tvecs)
        return ret, objp, corners2, err
    else:
        return ret, objp, None, None


def draw(img, corners, imgpts):
    # corner = tuple(corners[3].ravel()) #(0,0,0)
    corner = tuple(corners[-5].ravel()) #(0,0,0)
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,255,0), 5) #X (turquoise)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5) #Y
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5) #Z
    return img
