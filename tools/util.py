import math

import cv2 as cv
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import open3d
import tools.sgmf as sgmf

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


def unhom(a):
    if not isinstance(a, np.ndarray):
        aa = np.array(a)
    else:
        aa = a
    if aa[-1] == 0:
        return aa[:-1]
    return aa[:-1]/aa[-1]

def decimateRandom(pts1, pts2, proportion):
    n = min(pts1.shape[0], pts2.shapr[0])
    k = int(n * proportion)
    newPts1 = []
    newPts2 = []
    for i in range(0, k):
        r = random.randrange(0, n)
        newPts1.append(pts1[r])
        newPts2.append(pts2[r])

    if isinstance(pts1, np.ndarray):
        newPts1 = np.array(newPts1)
    if isinstance(pts2, np.ndarray):
        newPts2 = np.array(newPts2)

    return newPts1, newPts2

def decimate(pts1, pts2, proportion):
    n = min(pts1.shape[0], pts2.shapr[0])
    k = int(n * proportion)
    newPts1 = []
    newPts2 = []
    for i in range(0, k):
        r = random.randrange(0, n)
        newPts1.append(pts1[r])
        newPts2.append(pts2[r])

    if isinstance(pts1, np.ndarray):
        newPts1 = np.array(newPts1)
    if isinstance(pts2, np.ndarray):
        newPts2 = np.array(newPts2)

    return newPts1, newPts2

def transformSinglePoint3d(pt, mat):
    # pt: (3,1);  mat:(3,4)
    pt = np.append(pt, [1])
    mat = np.r_[mat, [[0,0,0,1]]]
    #print("pt", pt)
    #print("mat\n", mat)
    a = mat.dot(pt)
    #print("a", a)
    return unhom(a)

def transformPoints3d(pt, mat):
    # pt: (3,1);  mat:(3,4)
    if len(pt.shape) == 1:
        return None
    #pt = np.append(pt, [1])
    nbpts = pt.shape[0]
    #print("pt1", pt)
    pt = np.c_[pt, np.ones(nbpts)]
    mat = np.r_[mat, [[0,0,0,1]]]
    #print("pt2", pt)
    #print("mat\n", mat)
    a = np.transpose(mat.dot(np.transpose(pt)))
    #print("a\n", a)
    b = cv.convertPointsFromHomogeneous(a)
    b = b.reshape((b.shape[0], b.shape[-1]))
    #print("b\n", b)
    return b

def makeR4x4(R, T=None):
    if T is None:
        T = [0.0, 0.0, 0.0]
    M = np.c_[R, T]
    M = np.r_[M, [[0.0, 0.0, 0.0, 1.0]]]
    return M

def triangulate(campts, projpts, cameraIntrMatrix3x3, projIntrMatrix3x3, camDistCoeffs, projDistCoeffs, R, T):

    a = 0
    b = a+10
    print("campts\n", campts[a:b, :])
    print("projpts\n", projpts[a:b, :])

    undistCamPts = cv.undistortPoints(campts, cameraIntrMatrix3x3, camDistCoeffs)
    undistProjPts = cv.undistortPoints(projpts, projIntrMatrix3x3, projDistCoeffs)

    print("undistCamPts\n", undistCamPts[a:b, :])
    print("undistProjPts\n", undistProjPts[a:b, :])

    undistCamPts = undistCamPts.reshape((undistCamPts.shape[0], undistCamPts.shape[-1]))
    undistProjPts = undistProjPts.reshape((undistProjPts.shape[0], undistProjPts.shape[-1]))

    M1 = np.eye(3, 4, dtype=float)
    M2 = np.c_[R, T]
    """
    print("*******************************************")
    print("M1", M1.shape, "\n", M1)
    print("*******************************************")
    print("M2", M2.shape, "\n", M2)
    print("*******************************************")
    """
    reconpts4 = cv.triangulatePoints(M1, M2, np.transpose(undistCamPts), np.transpose(undistProjPts))
    reconpts4 = np.transpose(reconpts4)
    n = reconpts4.shape[0]

    reconpts = np.zeros((n, 3), dtype=float)
    for i in range(0, n):
        rp = unhom(reconpts4[i])
        reconpts[i] = rp

    print("undistCamPts", undistCamPts.shape)
    print("undistProjPts", undistProjPts.shape)
    print("reconpts", reconpts.shape)
    for i in range(0, 10):
        print(undistCamPts[i], " - ", undistProjPts[i], " - ", reconpts[i])

    return reconpts

def triangulateSGMF(sgmfPath, maskPath, texPath, projSize, cameraIntrMatrix3x3, projIntrMatrix3x3, camDistCoeffs, projDistCoeffs, R, T):

    mask = cv.imread(maskPath, cv.IMREAD_GRAYSCALE)
    tex = cv.imread(texPath, cv.IMREAD_GRAYSCALE)

    SGMF = sgmf.sgmf(sgmfPath, projSize, shadowMaskName=None)
    (camWidth, camHeight) = SGMF.camSize
    camPts = []
    projPts = []
    col = []
    i = 0

    a = 500000
    b = a + 10
    xx = 1250
    yy = 1020

    for y in range(0, camHeight):
        for x in range(0, camWidth):
            p = [x, y]
            q = SGMF.get_value(p)
            c = tex[y, x] / 255.0
            m = mask[y, x]
            if (x == xx) and y == yy:
                print(x, y, c, m)
            if m > 0:
                #camPts.append(p)
                #projPts.append(q)
                col.append([c,c,c])
                camPts.append([p[0] + 0.5, p[1] + 0.5])
                projPts.append([q[0] + 0.5, q[1] + 0.5])

                if a <= i < b:
                    print(p, " :: ", q, " :: ", c)
                i = i + 1



    camPts = np.array(camPts, dtype=float)
    projPts = np.array(projPts)

    pts3d = triangulate(camPts, projPts, cameraIntrMatrix3x3, projIntrMatrix3x3, camDistCoeffs, projDistCoeffs, R, T)

    return pts3d, np.array(col)

def removeBadPts(pts, cols, stdFactor=0.5, bbox=None):
    print("removeBadPts")
    print(pts[0:10, :])
    print(cols[0:10, :])

    avg = np.mean(pts, axis=0)
    std = np.std(pts, axis=0)
    print("avg", avg)
    print("std", std)
    print("min", np.min(pts, axis=0))
    print("max", np.max(pts, axis=0))
    minbox = avg-std*stdFactor
    maxbox = avg+std*stdFactor
    print("minbox", minbox)
    print("maxbox", maxbox)
    goodPts = []
    goodCols = []
    for k, p in enumerate(pts):
        yes = True
        for i in range(0, 3):
            if not minbox[i] < p[i] < maxbox[i]:
                yes = False
                break
            if bbox is not None:
                if not bbox[0][i] < p[i] < bbox[1][i]:
                    yes = False
                    break
        if yes:
            goodPts.append(p)
            c = cols[k]
            goodCols.append(c)

    return np.array(goodPts), np.array(goodCols)


def measureAligned(objpts, reconpts):
    n = objpts.shape[0]
    dists = []
    for i in range(0, n):
        op = objpts[i]
        rp = reconpts[i]
        d = np.linalg.norm(op-rp)
        dists.append(d)
        #print(op, "  -  ", rp, "  =  ", d)

    #print("dists:\n", np.array(dists))

    minDist = min(dists)
    maxDist = max(dists)
    avgDist = np.mean(dists)
    stdDist = np.std(dists)

    maxIdx = dists.index(maxDist)
    print("max idx", maxIdx)
    print(maxDist)
    print(objpts[maxIdx])
    print(reconpts[maxIdx])
    print(reconpts[maxIdx] - objpts[maxIdx])

    print("minDist", minDist)
    print("maxDist", maxDist)
    print("avgDist", avgDist)
    print("stdDist", stdDist)

    return minDist, maxDist, avgDist, stdDist


def measureSimDist(objpts, reconpts):
    """
    print("*******************************************")
    print("objpts", objpts.shape, "\n", objpts)
    print("*******************************************")
    print("reconpts", reconpts.shape, "\n", reconpts)
    print("*******************************************")
    """
    dists = []

    n = objpts.shape[0]
    """ only if aligned!!!!!!!!!
    for i in range(0, n):
        op = objpts[i]
        rp = reconpts[i]
        d = np.linalg.norm(op-rp)
        dists.append(d)
        #print(op, "  -  ", rp, "  =  ", d)

    minDist = min(dists)
    maxDist = max(dists)
    avgDist = np.mean(dists)
    stdDist = np.std(dists)

    print("minDist", minDist)
    print("maxDist", maxDist)
    print("avgDist", avgDist)
    print("stdDist", stdDist)
    """
    op0 = objpts[0]
    rp0 = reconpts[0]
    #print(op0)
    #print(rp0)
    distDiffs = []
    for i in range(1, n):
        #print("-----------------------------")
        op = objpts[i]
        rp = reconpts[i]
        #print(op)
        #print(rp)
        d1 = np.linalg.norm(op0 - op)
        d2 = np.linalg.norm(rp0 - rp)
        dd = abs(d1-d2)
        distDiffs.append(dd)
        #print(d1, " .. ", d2, " == ", dd)

    minDD = min(distDiffs)
    maxDD = max(distDiffs)
    avgDD = np.mean(distDiffs)
    stdDD = np.std(distDiffs)

    print("minDD", minDD)
    print("maxDD", maxDD)
    print("avgDD", avgDD)
    print("stdDD", stdDD)
    return minDD, maxDD, avgDD, stdDD


def listPlot(xData, yData, title, xLabel, yLabel, savePath=None, ySubLabels=None):
    ##x = np.linspace(0, 2, 100)
    # Note that even in the OO-style, we use `.pyplot.figure` to create the figure.
    fig, ax = plt.subplots()  # Create a figure and an axes.

    if not isinstance(yData, np.ndarray):
        yData = np.array(yData)

    nbx = len(xData)
    print("nbx",nbx)
    print("yshape", yData.shape)

    if len(yData.shape) < 2:
        ax.plot(xData, yData)
    else:

        if yData.shape[0] == nbx:
            yData = yData.transpose()
        elif yData.shape[1] == nbx:
            pass
        else:
            #error
            exit()

        nbData = yData.shape[0]
        print("nbData", nbData)
        for i in range(0, nbData):
            d = yData[i, :]
            lbl = "data" + str(i)
            if ySubLabels is not None:
                lbl = ySubLabels[i]
            print("d.shape", d.shape)
            ax.plot(xData, d, label=lbl)

    ##ax.plot(x, x, label='linear')  # Plot some data on the axes.
    ##ax.plot(x, x ** 2, label='quadratic')  # Plot more data on the axes...
    ##ax.plot(x, x ** 3, label='cubic')  # ... and some more.
    ax.set_xlabel(xLabel)  # Add an x-label to the axes.
    ax.set_ylabel(yLabel)  # Add a y-label to the axes.
    ax.set_title(title)  # Add a title to the axes.
    ax.legend()  # Add a legend.
    if savePath is None:
        plt.show()
    else:
        fig.savefig(savePath)


def calibrateExtrinsic(objp, imgp, projp, cameraMatrix, camDistCoeffs, projMatrix, projDistCoeffs):

    method = cv.SOLVEPNP_ITERATIVE
    if objp.shape[0] == 4:
        method = cv.SOLVEPNP_AP3P

    retval, rvec, camT = cv.solvePnP(objp, imgp, cameraMatrix, camDistCoeffs, flags=method)
    if math.isnan(np.sum(camT)) or not retval:
        #print("solvePnP cam: bad set of points!")
        raise Exception("bad set of points.  solvePnP retval=", retval, "camT=", camT)

    #print("retval", retval)
    #print("rvec", rvec)
    camR, _ = cv.Rodrigues(rvec)
    #print("camR:\n", camR)
    #print("camT", camT)

    retval, rvec, projT = cv.solvePnP(objp, projp, projMatrix, projDistCoeffs, flags=method)
    if math.isnan(np.sum(projT)) or not retval:
        #print("solvePnP proj: bad set of points!")
        raise Exception("bad set of points.  solvePnP retval=", retval, "projT=", projT)

    #print("retval", retval)
    #print("rvec", rvec)
    projR, _ = cv.Rodrigues(rvec)
    #print("projR:\n", projR)
    #print("projT", projT)

    cam4x4 = makeR4x4(camR, camT)
    proj4x4 = makeR4x4(projR, projT)
    iCam4x4 = np.linalg.inv(cam4x4)
    M = proj4x4.dot(iCam4x4)
    print("M=\n", M)

    R = M[0:3, 0:3]
    T = M[0:3, 3]
    #print("R=\n", R)
    #print("T=\n", T)
    return R, T
    """
     [[ 0.99750773 -0.01969474  0.06775279]
 [ 0.01730883  0.99921537  0.03562359]
 [-0.06840123 -0.03436208  0.99706596]]
T
 [[-0.61477313]
 [-0.00765998]
 [-0.28205569]]
    """


def normalizeND(pts):
    """
    # type: (np.ndarray) -> (np.ndarray, np.ndarray)

    :param point_list: point list to be normalized
    :return: normalization results

    m = np.mean(point_list[:2], axis=1)
    max_std = max(np.std(point_list[:2], axis=1)) + 1e-9
    c = np.diag([1 / max_std, 1 / max_std, 1])
    c[0][2] = -m[0] / max_std
    c[1][2] = -m[1] / max_std
    return np.dot(c, point_list), c

     m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1)) + 1e-9
    C1 = diag([1/maxstd, 1/maxstd, 1])
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    """
    #points are rows!
    nb = pts.shape[0]
    ndim = pts.shape[1]
    mean = np.mean(pts, axis=0)
    normMean = np.linalg.norm(mean)
    scale = normMean / math.sqrt(2.0)
    T = np.zeros((ndim+1, ndim+1))
    for i in range(0, ndim):
        T[i, i] = 1.0 / scale
    T[ndim, ndim] = 1.0
    for i in range(0, ndim):
        T[i, ndim] = -mean[i] / scale
    #print("T\n", T)
    hpts = np.c_[pts, np.ones(nb)]
    #print("hpts\n", hpts)
    npts = np.dot(T, hpts.transpose()).transpose()
    #print("npts\n", npts)
    #exit()
    return npts[:, 0:ndim], T


"""
def calculate_homography_matrix(origin, dest):

    assert origin.shape == dest.shape

    origin, c1 = normalize(origin)
    dest, c2 = normalize(dest)

    nbr_correspondences = origin.shape[1]
    a = np.zeros((2 * nbr_correspondences, 9))
    for i in range(nbr_correspondences):
        a[2 * i] = [-origin[0][i], -origin[1][i], -1, 0, 0, 0, dest[0][i] * origin[0][i], dest[0][i] * origin[1][i],
                    dest[0][i]]
        a[2 * i + 1] = [0, 0, 0, -origin[0][i], -origin[1][i], -1, dest[1][i] * origin[0][i], dest[1][i] * origin[1][i],
                        dest[1][i]]
    u, s, v = np.linalg.svd(a)
    homography_matrix = v[8].reshape((3, 3))
    homography_matrix = np.dot(np.linalg.inv(c2), np.dot(homography_matrix, c1))
    homography_matrix = homography_matrix / homography_matrix[2, 2]
    return homography_matrix
"""

def findHomography3D(pts1, pts2):

    if not isinstance(pts1, np.ndarray):
        pts1 = np.array(pts1)

    if not isinstance(pts2, np.ndarray):
        pts2 = np.array(pts2)

    assert pts1.shape == pts2.shape

    print("pts1:\n", pts1)
    print("pts2:\n", pts2)

    #pts1, c1 = normalizeND(pts1)
    #pts2, c2 = normalizeND(pts2)

    #print("*pts1:\n", pts1)
    #print("*pts2:\n", pts2)

    nb = pts1.shape[0]
    A = np.zeros((3*nb, 16))

    for i in range(0, nb):
        x1 = pts1[i][0]
        y1 = pts1[i][1]
        z1 = pts1[i][2]

        x2 = pts2[i][0]
        y2 = pts2[i][1]
        z2 = pts2[i][2]

        A[3 * i + 0] = [ -x1, -y1, -z1,  -1,   0,   0,   0,   0,   0,   0,   0,   0,  x2*x1, x2*y1, x2*z1, x2 ]
        A[3 * i + 1] = [ 0,   0,     0,   0, -x1, -y1, -z1,  -1,   0,   0,   0,   0,  y2*x1, y2*y1, y2*z1, y2 ]
        A[3 * i + 2] = [ 0,   0,     0,   0,   0,   0,   0,   0, -x1, -y1, -z1,  -1,  z2*x1, z2*y1, z2*z1, z2 ]

    U, S, V = np.linalg.svd(A)
    print("V.shape", V.shape)
    H = V[15].reshape((4, 4))
    #H = np.dot(np.linalg.inv(c2), np.dot(H, c1))
    H = H / H[3, 3]
    return H


def visualizePoints(arrayOfarraysOfPoints, arrayOfColors):

    if not isinstance(arrayOfarraysOfPoints, np.ndarray):
        arrayOfarraysOfPoints = np.array(arrayOfarraysOfPoints)

    if not isinstance(arrayOfColors, np.ndarray):
        arrayOfColors = np.array(arrayOfColors)

    assert arrayOfarraysOfPoints.shape[0] == arrayOfColors.shape[0]
    nb = arrayOfarraysOfPoints.shape[0]

    # open3d.visualization.draw_geometries([pcd1, pcd2])#, pcd3])

    visualizer = open3d.visualization.Visualizer()  # VisualizerWithEditing()
    # visualizer.create_window(window_name='Open3D', width=1920, height=1080, left=0, top=0, visible=True)
    visualizer.create_window()
    renderOptions = visualizer.get_render_option()
    renderOptions.background_color = np.asarray([0, 0, 0])
    renderOptions.point_size = 2
    renderOptions.show_coordinate_frame = True

    for i in range(0, nb):
        pts = arrayOfarraysOfPoints[i]
        col = np.array([arrayOfColors[i]] * pts.shape[0])
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(pts)
        pcd.colors = open3d.utility.Vector3dVector(col)
        #open3d.io.write_point_cloud("pcd"+str(i)+".ply", pcd)
        #pcds.append(pcd)
        visualizer.add_geometry(pcd)

    visualizer.run()
    visualizer.destroy_window()

def savePointsAsPly(path, points, color=None):

    if not isinstance(points, np.ndarray):
        points = np.array(points)

    if color is not None and not isinstance(color, np.ndarray):
        color = np.array(color)

    print("points.shape", points.shape)

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)

    if color is not None:
        print("color.shape", color.shape)
        #print(color)
        if color.shape[0] == 1:
            print("single color...")
            color = np.array([color[0]] * points.shape[0])
        print(color)
        print("min", np.min(color, axis=0))
        print("max", np.max(color, axis=0))
        pcd.colors = open3d.utility.Vector3dVector(color)

    open3d.io.write_point_cloud(path, pcd)


def measureold(objpts, imgpts1, imgpts2, K1, K2, coeffs1, coeffs2, R, T):
    print("*******************************************")
    print("objpts", objpts.shape, "\n", objpts)
    print("*******************************************")
    print("imgpts1", imgpts1.shape, "\n", imgpts1)
    print("*******************************************")
    print("imgpts2", imgpts2.shape, "\n", imgpts2)
    print("*******************************************")
    """
    M1 = np.c_[K1, [0.0, 0.0, 1.0]]
    print("*******************************************")
    print("**M1", M1.shape, "\n", M1)
    M1 = np.r_[M1, [[0.0, 0.0, 0.0, 1.0]]]
    print("*******************************************")
    print("**M1", M1.shape, "\n", M1)

    M2 = np.c_[K2, [0.0, 0.0, 0.0]]
    M2 = np.r_[M2, [[0.0, 0.0, 0.0, 1.0]]]

    RR = np.c_[R,  [0.0, 0.0, 0.0]]
    RR = np.r_[RR, [[0.0, 0.0, 0.0, 1.0]]]

    TT = np.eye(3, dtype=float)
    TT = np.c_[TT, T]
    TT = np.r_[TT, [[0.0, 0.0, 0.0, 1.0]]]

    M2 = M2.dot(RR).dot(TT)

    M1 = M1[:-1]
    M2 = M2[:-1]
    """

    n = objpts.shape[0]

    """
    M1 = np.eye(3, 4, dtype=float)
    M2 = np.c_[ R, T ]

    print("*******************************************")
    print("M1", M1.shape, "\n", M1)
    print("*******************************************")
    print("M2", M2.shape, "\n", M2)
    print("*******************************************")
    reconpts4 = cv.triangulatePoints(M1, M2, np.transpose(imgpts1), np.transpose(imgpts2))
    reconpts4 = np.transpose(reconpts4)
    reconpts = np.zeros((n,3), dtype=float)
    for i in range(0,n):
        rp = unhom(reconpts4[i])
        reconpts[i] = rp

    """
    reconpts = triangulate(imgpts1, imgpts2, K1, K2)
    print("reconpts", reconpts.shape, "\n", reconpts)
    print("*******************************************")

    dists = []

    for i in range(0, n):
        op = objpts[i]
        rp = reconpts[i]
        d = np.linalg.norm(op-rp)
        dists.append(d)
        #print(op, "  -  ", rp, "  =  ", d)

    minDist = min(dists)
    maxDist = max(dists)
    avgDist = np.mean(dists)
    stdDist = np.std(dists)

    print("minDist", minDist)
    print("maxDist", maxDist)
    print("avgDist", avgDist)
    print("stdDist", stdDist)

    op0 = objpts[0]
    rp0 = reconpts[0]
    #print(op0)
    #print(rp0)
    distDiffs = []
    for i in range(1, n):
        print("-----------------------------")
        op = objpts[i]
        rp = reconpts[i]
        #print(op)
        #print(rp)
        d1 = np.linalg.norm(op0 - op)
        d2 = np.linalg.norm(rp0 - rp)
        dd = abs(d1-d2)
        distDiffs.append(dd)
        print(d1, " .. ", d2, " == ", dd)

    minDD = min(distDiffs)
    maxDD = max(distDiffs)
    avgDD = np.mean(distDiffs)
    stdDD = np.std(distDiffs)

    print("minDD", minDD)
    print("maxDD", maxDD)
    print("avgDD", avgDD)
    print("stdDD", stdDD)
