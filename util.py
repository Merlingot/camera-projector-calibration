import cv2 as cv
import numpy as np
import os

def outputClean(output_paths):
    for path in output_paths:
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            for file in os.scandir(path):
                if file.name.endswith(".png"):
                    os.unlink(file.path)


def draw_reprojection(color, verifPath, objectPoints, imagePoints, cameraMatrix, distCoeffs, patternSize):
    def draw(img, origin, imgpts):
        # BGR
        img = cv.line(img, tuple(origin[0].ravel()), tuple(imgpts[0].ravel()), (255,0,0), 5) #X
        img = cv.line(img, tuple(origin[0].ravel()), tuple(imgpts[1].ravel()), (0,255,0), 5)
        img = cv.line(img, tuple(origin[0].ravel()), tuple(imgpts[2].ravel()), (0,255,255), 5) # Z en jaune
        return img

    objPoints=objectPoints.astype(np.float32)
    imgPoints=imagePoints.astype(np.float32)
    # Vérification de la calibration de la caméra en reprojection:
    # Montrer axes
    axis = np.float32([[1,0,0], [0,1,0], [0,0,1]]).reshape(-1,3)*1
    ret, rvecs, tvecs = cv.solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs)
    axisProj, jac = cv.projectPoints(axis, rvecs, tvecs, cameraMatrix, distCoeffs)
    origin = np.float32([[0,0,0]]).reshape(-1,1)
    originProj , jac = cv.projectPoints(origin, rvecs, tvecs, cameraMatrix, distCoeffs)
    img = draw(color.copy(), originProj[0], axisProj)
    cv.imwrite('{}reprojection_axes.png'.format(verifPath), img)
    pts, jac = cv.projectPoints(objPoints, rvecs, tvecs, cameraMatrix, distCoeffs)
    img = cv.drawChessboardCorners(color.copy(), patternSize, pts, 1)
    cv.imwrite('{}reprojection_cercles.png'.format(verifPath), img)


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
