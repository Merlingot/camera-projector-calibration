""" Calibration functions. See openCV documentation for ore information"""
import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt


# Nombre de coins internes du damier
NB_CORNER_WIDTH=5
NB_CORNER_HEIGHT=4
# Largeur d'un carré du damier
nb_pix = 48 # Largeur d'un carré en pixel
l_pix = 0.277e-3 # Largeur d'un pixel
squareSize = nb_pix*l_pix
FLIP=0 # flip par rapport à y


def takahashi(PATH, NB_CORNER_WIDTH, NB_CORNER_HEIGHT, squareSize, cam):
    """
    Finds and write corners coordinates for each images in PATH directory. Writes all information necessary to run Takahashi's algorithm after.
    Args:
        cam: 'AV' ou 'PG'
        PATH : path to images directory
        NB_CORNER_WIDTH : number of internal corners in the left/right direction
        NB_CORNER_HEIGHT : number of internal corners in the up/down direction
        squareSize : size of chessboard square in meter
    """
    CHECKERBOARD = ( NB_CORNER_WIDTH, NB_CORNER_HEIGHT)
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Images to analyse
    fnames = glob.glob(PATH)
    # Prepare object points (corner coordinates)
    objp = np.zeros((NB_CORNER_WIDTH*NB_CORNER_HEIGHT,3), np.float32)
    # objp[:,:2] = np.mgrid[0:NB_CORNER_WIDTH,0:NB_CORNER_HEIGHT].T.reshape(-1,2)

    k=0
    for j in range(NB_CORNER_HEIGHT):
        for i in range(NB_CORNER_WIDTH):
            objp[k] = [i,NB_CORNER_HEIGHT-j-1,0]
            k+=1

    objp = objp*squareSize

    if len(fnames) > 0:
        # Intrinsic parameters
        mtx, dist = intrinsic(NB_CORNER_WIDTH, NB_CORNER_HEIGHT, CHECKERBOARD, criteria, objp, fnames)
        # print(dist)
        # Write intrinsic parameter to file
        r=open("./data_{}/camera.txt".format(cam),"w+")
        for row in mtx :
            r.write("{} {} {}\n".format(row[0], row[1], row[2]))
        r.close()

        # Write intrinsic parameter to file
        r=open("./data_{}/dist.txt".format(cam),"w+")
        for row in dist :
            r.write("{} {} {} {} {}\n".format(row[0], row[1], row[2], row[3], row[4]))
        r.close()

        # Undistort and takahashi for each image
        index = [0, NB_CORNER_HEIGHT, -1] # Reference points
        j=1
        errs =[]; good_images=[];
        for fname in fnames:
            img = cv.imread(fname);
            # img = cv.flip(img, FLIP)
            dst = undistort(mtx, dist, img)
            ret, objpoints, imgpoints, err = find_corners(NB_CORNER_WIDTH, NB_CORNER_HEIGHT, CHECKERBOARD, criteria, objp, dst, mtx, dist)
            # Write corner image coordinates for this image
            if ret == True :
                errs.append(err)
                good_images.append(j)
                f=open("./data_{}/input{}.txt".format(cam, j),"w+")
                for i in index :
                    p = imgpoints[i]
                    f.write("{} {}\n".format(p[0,0], p[0,1]))
                f.close()
                j+=1

        # Write corner world coordinates
        m=open("./data_{}/model.txt".format(cam),"w+")
        for i in index :
            p = objpoints[i]
            m.write("{} {} {}\n".format(p[0], p[1], p[2]))
        m.close()

        return errs, good_images
    else :
        print('No images found')


def intrinsic(NB_CORNER_WIDTH, NB_CORNER_HEIGHT, CHECKERBOARD, criteria, objp, fnames):
    """
    Find the intrinsic calibration parameters (camera matrix and distortion coefficient) of one camera
    Args:
        NB_CORNER_WIDTH : number of internal corners in the left/right direction
        NB_CORNER_HEIGHT : number of internal corners in the up/down direction
        squareSize : size of chessboard square in meter
        fnames : path to images to analyses
    Returns:
        mtx : camera matrix (K)
        dist : distortion coefficient
    """
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for fname in fnames:
        img = cv.imread(fname);
        # img = cv.flip(img, FLIP)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners ----------------------------
            cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            cv.imwrite('corner.png', cv.flip(img, FLIP))
            # cv.imshow('img', img)
            # cv.waitKey(0)
    cv.destroyAllWindows()
    # ----------------------------
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Montrer axes ----------------------------------
    axis = np.float32([[3,0,0], [0,3,0], [0,0,3]]).reshape(-1,3)*l_pix*nb_pix
    for fname in fnames:
        img = cv.imread(fname);
        # img = cv.flip(img, FLIP)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD,None)
        if ret == True:
            corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            # Find the rotation and translation vectors.
            ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
            # project 3D points to image plane
            imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
            img = draw(img,corners2,imgpts)
            cv.imwrite('axis.png', cv.flip(img, FLIP))
            # cv.imshow('img',img)
            # cv.waitKey(0)
    cv.destroyAllWindows()

    return mtx, dist

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


# Code
PATH = '/Users/mariannelado-roy/projet4/Calibration_03_13_2020/ext_manta/*.png'
errs, good_images = takahashi(PATH, NB_CORNER_WIDTH, NB_CORNER_HEIGHT, squareSize, 'AV')

plt.figure()
plt.title('Erreur de reprojection AV')
plt.xlabel('Image #')
plt.ylabel('Erreur (pixel)')
plt.bar(good_images, errs)
plt.savefig('./data_AV/err_AV.png', format='png')


PATH = '/Users/mariannelado-roy/projet4/Calibration_03_13_2020/ext_pg/*.png'
errs, good_images = takahashi(PATH, NB_CORNER_WIDTH, NB_CORNER_HEIGHT, squareSize, 'PG')

plt.figure()
plt.title('Erreur de reprojection AV')
plt.xlabel('Image #')
plt.ylabel('Erreur (pixel)')
plt.bar(good_images, errs)
plt.savefig('./data_PG/err_PG.png', format='png')
