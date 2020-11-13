""" Tsai calibration """

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

import util

dataPath="data/serie_1/"
noFringePath=os.path.join(dataPath, "nofringe/noFringe.png")
color=cv.imread(noFringePath)
gray = cv.cvtColor(color , cv.COLOR_BGR2GRAY)
imageShape=gray.shape
pointsPath=os.path.join(dataPath,"points/")
outputPath=os.path.join(dataPath,"tsai/")
util.outputClean([outputPath])
imageShape
# Parametres -------------


# STAGE 1 ) Compute 3D Orientation, Position and Scale Factor:
# a) Calculate distorted image coordinates (Xd, Yd)
# i ) Xf Yf
data = np.loadtxt(pointsPath+"points.txt")
n=data.shape[0]
points3d = data[:, :3]
points2d = data[:, 3:]
xw=points3d[:,0]; yw=points3d[:,1]; zw=points3d[:,2];
Xf=points2d[:,0]; Yf=points2d[:,1]

# plt.figure()
# plt.title('Centres des cercles détectés')
# plt.xlabel('X (pixels - computer frame memory)')
# plt.ylabel('Y (pixels - computer frame memory)')
# plt.plot(Xf,Yf, 'o')
# plt.savefig( "{}computerFrame.png".format(outputPath) )
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_title('Centres des cercles donnés')
# ax.set_xlabel('xw (m - world coord)')
# ax.set_ylabel('yw (m - world coord)')
# ax.set_zlabel('zw (m - world coord)')
# ax.scatter( xw, yw, zw, zdir='z')
# plt.savefig( "{}worldCoord.png".format(outputPath) )

# ii) Manta Allied Vision
sx=1
dx=3.75e-6 #Center to center distance between adjacent sensor element in X
dy=3.75e-6 #Center to center distance between adjacent sensor element in Y
Ncx=1   #Number of sensor element in X direction
Nfx= 1  #Number of pixel in a line as sampled by the computer ????
dxx=dx*Ncx/Nfx
# iii)
#Row ans column number of center of computer frame memory
Cx=imageShape[0];Cy=imageShape[1];
# iv) Compute Xdi Ydi
Xd=sx**(-1)*dxx*(Xf-Cx)
Yd=dy*(Yf-Cy)

## b) Compute the seven unknowns
# Contruire la matrice A
A = np.zeros((n,7))
for i in range(n):
    A[i,:]=np.array([ Yd[i]*xw[i], Yd[i]*yw[i], Yd[i]*zw[i], Yd[i], -Xd[i]*xw[i], -Xd[i]*yw[i], -Xd[i]*zw[i] ]  )
# Vecteur b:
b=Xd
# Solve Ax=b avec décomposition QR
Q,R = np.linalg.qr(A) # qr decomposition of A
Qb = np.dot(Q.T,b) # computing Q^T*b (project b onto the range of A)
sol= np.linalg.solve(R,Qb) # solving R*x = Q^T*b

## c) Seven unknows
a1,a2,a3,a4,a5,a6,a7 = sol[0],sol[1],sol[2],sol[3],sol[4],sol[5],sol[6]
# 1)Compute |Ty|
normTy = (a5**2 + a6**2 + a7**2)**(-1/2)
# 2)Determine the sign of Ty
# i) pick an image point away from C -> coin (0,0,0)
Xfi,Yfi=Yf[0],Yf[0]
xwi,ywi,zwi=xw[0],yw[0],zw[0]
# ii) pick the sign of Ty to be 1
Ty=1*normTy
# iii) Compute
r1=a1*Ty; r2=a2*Ty
r4=a5*Ty
r5=a6*Ty; Tx=a4*Ty
xi=r1*xwi+r2*ywi+Tx; yi=r4*xwi+r5*ywi+Ty
# iv)
if not (xi*Xfi>0 and yi*Yfi>0):
    Ty = normTy*(-1)
# 3) Determine sx:
sx=(a1**2+a2**2+a3**2)**(1/2)*normTy
# 4) Compute the 3D rotation matrix R
r1=a1*Ty/sx
r2=a2*Ty/sx
r3=a3*Ty/sx
r4=a4*Ty
r5=a6*Ty
r6=a7*Ty
Tx=a4*Ty

R = np.zeros((3,3))
row1=np.array([r1, r2, r3])
row2=np.array([r4, r5, r6])
row3=np.cross(row1, row2)
R[0,:]=row1; R[1,:]=row2; R[2,:]=row3
r7,r8,r9=row3[0],row3[1],row3[2]
det=np.linalg.det(R)


# STAGE 2)
Y=dy**(-1)*Yd + Cy
X = Xd*Nfx/(dx*Ncx)
# a) compute an approximation of f and Tz
A = np.zeros((n,2))
b = np.zeros(n)
for i in range(n):
    yi = r4*xw[i]+r5*yw[i]+r6*0
    wi=r7*xw[i]+r8*yw[i]+r9*0
    A[i,:]=np.array([ yi, -dy*Y[i] ])
    b[i]=wi*dy*Y[i]

sol=np.linalg.lstsq(A,b, rcond=None)
f,Tx = sol[0],sol[1]








#
