import cv2 as cv
import numpy as np
import os


def clean_folders(output_paths, ext=".png"):
    for path in output_paths:
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            for file in os.scandir(path):
                if file.name.endswith(ext):
                    os.unlink(file.path)


def coins_damier(patternSize,squaresize):
    objp = np.zeros((patternSize[0]*patternSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1, 2)
    objp*=squaresize
    return objp

def draw_reprojection(color, verifPath, objectPoints, imagePoints, cameraMatrix, distCoeffs, patternSize, squaresize=10e-2):

    objPoints=objectPoints.astype(np.float32)
    imgPoints=imagePoints.astype(np.float32)

    ret, rvecs, tvecs = cv.solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs)
    img	=cv.drawFrameAxes(color.copy(), cameraMatrix, distCoeffs, rvecs, tvecs, squaresize, 5)
    cv.imwrite('{}reprojection_axes.png'.format(verifPath), img)

    pts, jac = cv.projectPoints(objPoints, rvecs, tvecs, cameraMatrix, distCoeffs)
    img = cv.drawChessboardCorners(color.copy(), patternSize, pts, 1)
    cv.imwrite('{}reprojection_points.png'.format(verifPath), img)
    return img

def outputClean(output_paths):
    for path in output_paths:
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            for file in os.scandir(path):
                if file.name.endswith(".png"):
                    os.unlink(file.path)





def intrinsic_matrix(f, Cx, Cy, dx, dy):
    mx = 1/dx; my=1/dy
    return np.array([ [f*mx, 0, Cx],[0,f*my,Cy],[0,0,1] ])

def reprojection_err(imagePoints, projectedPoints):
    err = np.sum(( imagePoints[:,0,0] -  projectedPoints[:,0,0] )**2 ) + np.sum( ( imagePoints[:,0,1] -  projectedPoints[:,0,1] )**2)
    n=imagePoints.shape[0]
    return np.sqrt(err/n)


def formatage(n, world, computerImage, Cx, Cy, dx, dy, R, T, sx, f_, Tz_, k1_, k2_):
    # preparer les données pour project points
    objectPoints = np.zeros((n,3))
    imagePoints = np.zeros((n,1,2))
    for i in range(n):
        objectPoints[i,:] = np.array([world[0][i], world[1][i], world[2][i]])
        imagePoints[i,0,:] = np.array([computerImage[0][i], computerImage[1][i]])
    # Rigid transform:
    rvec, _ = cv.Rodrigues(R)
    tvec = np.array([T[0],T[1],Tz_])
    # Intrinsic Matrix :
    cameraMatrix = intrinsic_matrix(f_, Cx, Cy, dx,dy)
    # Distortion coefficients
    distCoeffs = np.array([k1_, k2_,0,0])
    return imagePoints, objectPoints, rvec, tvec, cameraMatrix, distCoeffs
