
import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
import sgmf
# from tsai import calibrate
from blob import detect_centers
from circlesgrid import getAsymCirclesObjPoints

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


plt.figure()
plt.title('Coté droit')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, np.max(objp1[:,0])*1.2)
plt.plot(objp1[:,0], objp1[:,1],'o-')
plt.plot(objp1[:,0][0], objp1[:,1][0],'o', color='C1')

fig, ax = plt.subplots()
plt.title('Coté gauche')
plt.xlabel('z')
plt.ylabel('y')
ax.set_xlim(np.max(objp2[:,2])*1.2, 0)
plt.plot(objp2[:,2], objp2[:,1],'o-')
plt.plot(objp2[:,2][0], objp2[:,1][0],'o', color='C1')

plt.figure()
plt.title('Damier complet vue de haut')
plt.xlabel('x')
plt.ylabel('z')
plt.plot(objp[:,0], objp[:,2],'o')


# Nom des images dans une liste
imgPath="data/serie_1/nofringe/*.png"
fnames = glob.glob(imgPath)

if (len(fnames)>0) :
    # lire les images
    color=cv.imread(fnames[0])
    gray = cv.cvtColor(color , cv.COLOR_BGR2GRAY)

plt.imshow(color)

detect_centers(patternSize, objp, color, gray)
