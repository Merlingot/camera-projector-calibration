""" Fichier pour détecter les centres des cercles et écrire les coordonées dans un fichier .txt"""
import os
import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt

# Personal imports :
import sgmf
from detectCenters import detect_centers
from circlesgrid import getAsymCirclesObjPoints
import util

# Data:
dataPath="data/serie_1/"

verifPath=os.path.join(dataPath,"verif/")
pointsPath=os.path.join(dataPath,"points/")
output_paths=[verifPath, pointsPath]
util.outputClean(output_paths)


#Résolution du projecteur en pixel
projSize=(800,600)

# Damier
points_per_row=4; points_per_colum=11
patternSize = (points_per_row, points_per_colum)
realPatternSize=(2*points_per_row, points_per_colum)
circleDiameter=1.5e-2
circleSpacing=2e-2
paperMargin = 3e-2 #À trouver

# Points 3d
objp1 = getAsymCirclesObjPoints(points_per_colum, points_per_row, circleDiameter+circleSpacing, paperMargin, 0, "xy")
objp2 = getAsymCirclesObjPoints(points_per_colum, points_per_row, circleDiameter+circleSpacing, paperMargin, 0, "yz")
objp=np.concatenate((objp1, objp2))

# Vérification des points 3D
fig, ax = plt.subplots()
plt.title('Coté droit')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.xlim(0, np.max(objp1[:,0])*1.2)
plt.plot(objp1[:,0], objp1[:,1],'o-')
plt.plot(objp1[:,0][0], objp1[:,1][0],'o', color='C1')
plt.savefig('{}damier_droit.png'.format(verifPath), format="png")

fig, ax = plt.subplots()
plt.plot(objp2[:,2], objp2[:,1],'o-')
plt.plot(objp2[:,2][0], objp2[:,1][0],'o', color='C1')
plt.title('Coté gauche')
plt.xlabel('z (m)')
plt.ylabel('y (m)')
ax.set_xlim(np.max(objp2[:,2])*1.2, 0)
plt.savefig('{}damier_gauche.png'.format(verifPath), format="png")

fig, ax = plt.subplots()
ax.set_title('Damier complet vue de haut')
ax.set_xlabel('x (m)')
ax.set_ylabel('z (m)')
ax.plot(objp[:,0], objp[:,2],'o')
plt.savefig('{}damier_vue_haut.png'.format(verifPath), format="png")


# Nom des images dans une liste
noFringePath=os.path.join(dataPath, "nofringe/*.png")
fnames = glob.glob(noFringePath)
if (len(fnames)>0) :
    color=cv.imread(fnames[0])
    gray = cv.cvtColor(color , cv.COLOR_BGR2GRAY)

imgPoints=detect_centers(patternSize, objp, color, gray, verifPath, pointsPath)
