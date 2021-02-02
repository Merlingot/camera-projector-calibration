""" calibration Zhang pour intrinsèques et avec les cibles pour extrinsèques """
import math

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import glob
import random
from tools.findPoints import proj_centers
from tools.util import coins_damier, clean_folders, draw_reprojection, measureSimDist, measureAligned, triangulate, transformPoints3d, calibrateExtrinsic, listPlot
"""
print(len(np.array([1,2,4]).shape))

m = [[2,0,0,1],[0,2,0,1],[0,0,2,1]]
m = np.array(m)
p = np.array([1,2,3])
a = transformSinglePoint3d(p, m)
print(a)

pp = np.array([[1,2,3],[3,2,1]])
b = transformPoints3d(pp, m)
print(b)
exit()
"""
# Data ========================================================================
dataPath="data/14_01_2021/"
# Data pour intrinsèque:
#noFringePath=np.sort(glob.glob(os.path.join(dataPath,"zhang/max*.png")))
#sgmfPath=np.sort(glob.glob(os.path.join(dataPath,"zhang/match*.png")))
noFringePath=np.sort(glob.glob("/home/lbouchard/data/crvi/14jan/zhang/*/max_00.png"))
sgmfPath=np.sort(glob.glob("/home/lbouchard/data/crvi/14jan/zhang/*/match_00.png"))
# Data pour extrinsèque:
objpRT_path=os.path.join(dataPath,"RT-calib/objpts.txt")
imgpRT_path=os.path.join(dataPath,"RT-calib/imgpts.txt")
sgmfRT_path=os.path.join(dataPath,"RT-calib/match_00.png")

# Output:
outputPath=os.path.join(dataPath,"output2/")
imagesPath=os.path.join(outputPath,"detection/")
outputfile=os.path.join(outputPath,'erreur.txt')
# Créer/Vider les folder/file:
clean_folders([outputPath,imagesPath], ".png")
# ==============================================================================

# Paramètres ===================================================================
# Camera:
imageSize=(2464, 2056)
# Projecteur:
projSize=(1920,1200)
# Damier
points_per_row=10; points_per_colum=7
squaresize=10e-2
patternSize=(points_per_row, points_per_colum)
# ==============================================================================

# ==== LIRE LES POINTS =========================================================

# Damier -----------------------------------------------------------------------
# Listes de points
objectPoints=[]; imagePoints=[]; projPoints=[]
# Points 3d
objp = coins_damier(patternSize, squaresize)
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for i in range(len(noFringePath)):
    # Lire l'image
    color=cv.imread(noFringePath[i])
    gray=cv.cvtColor(color, cv.COLOR_BGR2GRAY)
    # Find corners
    ret, corners = cv.findChessboardCorners(gray, patternSize, None)
    if ret==True:
        imgp = cv.cornerSubPix(gray, corners, patternSize, (-1,-1), criteria)
        # Ajouter aux listes
        objectPoints.append(objp)
        imagePoints.append(imgp)
        # Points du projecteur
        projp = proj_centers(objp, imgp, projSize, sgmfPath[i])
        projPoints.append(projp)
        # Enregistrer les coins détectés - optionnel
        img = cv.drawChessboardCorners(color.copy(), patternSize, imgp, ret)
        cv.imwrite("{}corners_{}.png".format(imagesPath,i),img)
# ------------------------------------------------------------------------------

# Cibles -----------------------------------------------------------------------
# Lire les points
objpRT=np.genfromtxt(objpRT_path).astype(np.float32)
# offset x,y
objpRT[:,1]-=objpRT[2,1]
objpRT[:,0]-=objpRT[2,0]

imgpRT=np.genfromtxt(imgpRT_path).astype(np.float32)
imgpRT=imgpRT.reshape(24, 1, 2)
# Points du projecteur
projpRT = proj_centers(objpRT, imgpRT, projSize, sgmfRT_path)
# ------------------------------------------------------------------------------

# Choix de points pour la calibration intrinsèque:
objectPoints_=objectPoints.copy()
imagePoints_=imagePoints.copy()
projPoints_=projPoints.copy()
objectPoints_.append(objpRT)
imagePoints_.append(imgpRT)
projPoints_.append(projpRT)

#===============================================================================
#==== CALIBRATION INTRINSÈQUE ==================================================
# Camera -----------------------------------------------------------------------
retC, cameraMatrix, camDistCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors=cv.calibrateCameraExtended(objectPoints_, imagePoints_, imageSize, np.zeros((3,3)), np.zeros((1,4)))
# ------------------------------------------------------------------------------

# Projecteur -------------------------------------------------------------------
retP, projMatrix, projDistCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors=cv.calibrateCameraExtended(objectPoints_, projPoints_, projSize, np.zeros((3,3)), np.zeros((1,4)))
# ------------------------------------------------------------------------------
#===============================================================================

#==== CALIBRATION EXTRINSÈQUE ==================================================
retval, cameraMatrix, camDistCoeffs, projMatrix, projDistCoeffs, R, T, _, _, perViewErrors = cv.stereoCalibrateExtended([objpRT], [imgpRT], [projpRT], cameraMatrix, camDistCoeffs, projMatrix, projDistCoeffs, imageSize, None, None, flags=cv.CALIB_FIX_INTRINSIC)
#print("R\n", R)
#print("T\n", T)
#calibrateExtrinsic(objpRT, imgpRT, projpRT, cameraMatrix, camDistCoeffs, projMatrix, projDistCoeffs)
originalT = T.flatten()
originalR = R.copy()
#===============================================================================
color=cv.imread("data/14_01_2021/RT-calib/cornerSubPix.png")
img =draw_reprojection(color, outputPath, objpRT, imgpRT, cameraMatrix, camDistCoeffs, (4,6), 1)

print("##################################")
print("########  MEASUREMENTS  ##########")
print("##################################")

reconpts = triangulate(imgpRT, projpRT, cameraMatrix, projMatrix, camDistCoeffs, projDistCoeffs, R, T)
print("reconpts\n", reconpts)
measureSimDist(objpRT, reconpts)
#solve...
#retval, rvec, tvec	=	cv.solvePnP(	objectPoints, imagePoints, cameraMatrix, distCoeffs[, rvec[, tvec[, useExtrinsicGuess[, flags]]]]	)
#retval, rvecs, tvecs, reprojectionError	=	cv.solvePnPGeneric(	objectPoints, imagePoints, cameraMatrix, distCoeffs[, rvecs[, tvecs[, useExtrinsicGuess[, flags[, rvec[, tvec[, reprojectionError]]]]]]]
#retval, rvec, tvec, inliers	=	cv.solvePnPRansac(	objectPoints, imagePoints, cameraMatrix, distCoeffs[, rvec[, tvec[, useExtrinsicGuess[, iterationsCount[, reprojectionError[, confidence[, inliers[, flags]]]]]]]]	)
#refine...
#rvec, tvec = cv.solvePnPRefineLM(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec[, criteria]    )
#rvec, tvec	=	cv.solvePnPRefineVVS(	objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec[, criteria[, VVSlambda]]	)

retval, rvec, tvec = cv.solvePnP(objpRT, imgpRT, cameraMatrix, camDistCoeffs)

print("retval", retval)
print("rvec", rvec)
R, _ = cv.Rodrigues(rvec)
print("R:\n", R)
print("tvec", tvec)
RT = np.c_[R, tvec]
print("RT:\n", RT)

transpoints = transformPoints3d(objpRT, RT)
print("transpoints\n", transpoints)
measureAligned(reconpts, transpoints)
#exit()

print("##################################")
print("##########  DECIMATION  ##########")
print("##################################")
###nb = 100
repeats = 1000

simMinDistData = []
simMaxDistData = []
simAvgDistData = []
simStdDistData = []

alignedMinDistData = []
alignedMaxDistData = []
alignedAvgDistData = []
alignedStdDistData = []

avgSimMinDistdata = []
avgSimMaxDistdata = []
avgSimAvgDistdata = []
avgSimStdDistdata = []

avgAlignedMinDistData = []
avgAlignedMaxDistData = []
avgAlignedAvgDistData = []
avgAlignedStdDistData = []

nbPicks = []

for i in range(4, objpRT.shape[0]):
    #n = i
#for i in range(0, nb):

    simMinDistData.append([])
    simMaxDistData.append([])
    simAvgDistData.append([])
    simStdDistData.append([])

    alignedMinDistData.append([])
    alignedMaxDistData.append([])
    alignedAvgDistData.append([])
    alignedStdDistData.append([])

    #u = 0.3
    #v = u + (1.0 - u) * i / nb
    #n = int(objpRT.shape[0] * v)

    for j in range(repeats):
        print(j)
        #picks = [random.randint(0, objpRT.shape[0]-1) for i in range(n)] # randint inclusive!!.. but bad.. duplicates!
        success = False
        while not success:
            picks = np.random.choice(range(0, objpRT.shape[0]-1), i, replace=False) #replace false -> no dupl
            print("picks", picks)
            objp = objpRT[picks]
            imgp = imgpRT[picks]
            projp = projpRT[picks]
            try:
                if len(picks) == 4:
                    R, T = calibrateExtrinsic(objp, imgp, projp, cameraMatrix, camDistCoeffs, projMatrix, projDistCoeffs)
                    print("R=\n", R)
                    print("T=\n", T)
                else:
                    retval, cameraMatrix, camDistCoeffs, projMatrix, projDistCoeffs, R, T, _, _, perViewErrors = cv.stereoCalibrateExtended([objp], [imgp], [projp], cameraMatrix, camDistCoeffs, projMatrix, projDistCoeffs, imageSize, None, None, flags=cv.CALIB_FIX_INTRINSIC)
                    print("retval", retval)
                    print("perViewErrors", perViewErrors)
            except Exception as e:
                print('Caught this error: ' + repr(e))
                print("retry with new pts")
                # exit()
                continue

            tdist = np.linalg.norm(T.flatten() - originalT)
            rdist = np.linalg.norm(R - originalR)
            if tdist > 0.05:
                print("T", T)
                print("originalT", originalT)
                print("diff", T.flatten() - originalT)
                print("normdiff", tdist)
                print("bad T: retry with new pts")
                # exit()
                continue
            if rdist > 0.002:
                print("R", R)
                print("originalR", originalR)
                print("diff", R - originalR)
                print("normdiff", rdist)
                print("bad R: retry with new pts")
                # exit()
                continue
            success = True

        reconpts = triangulate(imgpRT, projpRT, cameraMatrix, projMatrix, camDistCoeffs, projDistCoeffs, R, T)

        mind, maxd, avgd, stdd = measureSimDist(objpRT, reconpts)
        if math.isnan(mind) or math.isnan(maxd) or math.isnan(avgd) or math.isnan(stdd):
            print("objp", objp)
            print("imgp", imgp)
            print("projp", projp)
            print(reconpts)
            print(mind, maxd, avgd, stdd)
            print(R)
            print(T)
            exit()
        #simDistdata[-1].append(stdd)
        simMinDistData[-1].append(mind)
        simMaxDistData[-1].append(maxd)
        simAvgDistData[-1].append(avgd)
        simStdDistData[-1].append(stdd)
        """
        if maxd > 1.0:
            print("bad maxd = ", maxd)
            print("R\n", R)
            print("originalR\n", originalR)
            print("diff", np.linalg.norm(R-originalR))
            print("T\n", T)
            print("originalT\n", originalT)
            print("diff", np.linalg.norm(T - originalT))
            exit()
        """
        mind, maxd, avgd, stdd = measureAligned(reconpts, transpoints)
        #alignedDistData[-1].append(stdd)
        alignedMinDistData[-1].append(mind)
        alignedMaxDistData[-1].append(maxd)
        alignedAvgDistData[-1].append(avgd)
        alignedStdDistData[-1].append(stdd)
        print("-----------------------------------------")

    nbPicks.append(i)

    a = np.mean(np.array(simMinDistData[-1]))
    avgSimMinDistdata.append(a)
    a = np.mean(np.array(simMaxDistData[-1]))
    avgSimMaxDistdata.append(a)
    a = np.mean(np.array(simAvgDistData[-1]))
    avgSimAvgDistdata.append(a)
    a = np.mean(np.array(simStdDistData[-1]))
    avgSimStdDistdata.append(a)

    a = np.mean(np.array(alignedMinDistData[-1]))
    avgAlignedMinDistData.append(a)
    a = np.mean(np.array(alignedMaxDistData[-1]))
    avgAlignedMaxDistData.append(a)
    a = np.mean(np.array(alignedAvgDistData[-1]))
    avgAlignedAvgDistData.append(a)
    a = np.mean(np.array(alignedStdDistData[-1]))
    avgAlignedStdDistData.append(a)

print("##################################")
print("##################################")
np.set_printoptions(suppress=True)

print("nbpicks", nbPicks)
#print("simDistdata", np.c_[nbPicks, avgSimDistdata, stdSimDistdata])
#print("alignedDistData", np.c_[nbPicks, avgAlignedDistData, stdAlignedDistData])

ylbls = ["min", "max", "avg", "std"]
ydata = np.c_[avgSimMinDistdata, avgSimMaxDistdata, avgSimAvgDistdata, avgSimStdDistdata]
listPlot(nbPicks, ydata, "Similar distances", "Number of targets", "Distance (m)", "./sim_plt1.png", ySubLabels=ylbls)
ydata = np.c_[avgAlignedMinDistData, avgAlignedMaxDistData, avgAlignedAvgDistData, avgAlignedStdDistData]
listPlot(nbPicks, ydata, "Aligned target distances", "Number of targets", "Distance (m)", "./aligned_plt1.png", ySubLabels=ylbls)

print("avgSimMinDistdata", avgSimMinDistdata)
print("avgSimMaxDistdata", avgSimMaxDistdata)
print("avgSimAvgDistdata", avgSimAvgDistdata)
print("avgSimStdDistdata", avgSimStdDistdata)

print("avgAlignedMinDistData", avgAlignedMinDistData)
print("avgAlignedMaxDistData", avgAlignedMaxDistData)
print("avgAlignedAvgDistData", avgAlignedAvgDistData)
print("avgAlignedStdDistData", avgAlignedStdDistData)

print("###################### 4 corners ########################")
print(objpRT)
picks = [0, 3, 19, 23]
print(np.array(objpRT)[picks])
objp = objpRT[picks]
imgp = imgpRT[picks]
projp = projpRT[picks]

R, T = calibrateExtrinsic(objp, imgp, projp, cameraMatrix, camDistCoeffs, projMatrix, projDistCoeffs)
print("R=\n", R)
print("T=\n", T)
reconpts = triangulate(imgpRT, projpRT, cameraMatrix, projMatrix, camDistCoeffs, projDistCoeffs, R, T)
mind, maxd, avgd, stdd = measureSimDist(objpRT, reconpts)
print("measureSimDist:")
print("mind", mind)
print("maxd", maxd)
print("avgd", avgd)
print("stdd", stdd)
mind, maxd, avgd, stdd = measureAligned(reconpts, transpoints)
print("measureAligned:")
print("mind", mind)
print("maxd", maxd)
print("avgd", avgd)
print("stdd", stdd)


"""
x=[0,1,2,3,4,5,6,7]
y=[[9,8,7,6,5,4,3,2], [3,4,3,4,3,4,3,4]]
ylbls = ["aaaa", "bbbb"]
listPlot(x, y, "title", "x", "y", "./plt.png", ySubLabels=ylbls)
"""

#print("avgAlignedDistData", avgAlignedDistData)
#print("stdSimDistdata", stdSimDistdata)
#print("stdAlignedDistData", stdAlignedDistData)

###################################################################################
# measurements!
#print("*******************************************")
#print("cameraMatrix", cameraMatrix.shape, "\n", cameraMatrix)
#print("*******************************************")
#print("projMatrix", projMatrix.shape, "\n", projMatrix)
#print("*******************************************")
"""
undistCamPts = cv.undistortPoints(imgpRT, cameraMatrix, camDistCoeffs)
undistProjPts = cv.undistortPoints(projpRT, projMatrix, projDistCoeffs)

undistCamPts = undistCamPts.reshape((undistCamPts.shape[0], undistCamPts.shape[-1]))
undistProjPts = undistProjPts.reshape((undistProjPts.shape[0], undistProjPts.shape[-1]))

measure(objpRT, undistCamPts, undistProjPts, cameraMatrix, projMatrix, R, T)

"""



#==== ENREGISTRER ==============================================================
# Enregistrer:
s = cv.FileStorage()
s.open('{}cam.xml'.format(outputPath), cv.FileStorage_WRITE)
s.write('K',cameraMatrix)
s.write('R', np.eye(3))
s.write('t', np.zeros(T.shape))
s.write('coeffs', camDistCoeffs)
s.write('imageSize', imageSize)
s.release()

# Enregistrer:
s = cv.FileStorage()
s.open('{}proj.xml'.format(outputPath), cv.FileStorage_WRITE)
s.write('K', projMatrix)
s.write('R', R)
s.write('t', T)
s.write('coeffs', projDistCoeffs)
s.write('imageSize', projSize)
s.release()
#===============================================================================
f=open(outputfile, 'w+'); f.close()
f=open(outputfile, 'a')
f.write('- Calibration Stéréo Zhang - \n \n')
f.write('Erreur de reprojection RMS caméra:\n')
f.write("{}\n".format(retC))
f.write('Erreur de reprojection RMS projecteur:\n')
f.write("{}\n".format(retP))
f.write('Erreur de reprojection RMS stéréo:\n')
f.write("{}\n".format(retval))
f.close()

print("done")

