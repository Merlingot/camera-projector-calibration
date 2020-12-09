""" Tsai calibration linéaire (stage 1) """

import numpy as np
import cv2 as cv
from math import atan2, sin, cos, asin


# Parametres -------------
class Params():
    def __init__(self, imageShape,pixelSize):
        self.sx=1
        self.dx=pixelSize[0] #Center to center distance between adjacent sensor element in X
        self.dy=pixelSize[1]#Center to center distance between adjacent sensor element in Y
        self.Ncx=imageShape[0]   #Number of sensor element in X direction
        self.Nfx=imageShape[0]   #Number of pixel in a line as sampled by the computer
        self.dxx=self.dx*self.Ncx/self.Nfx
        #Row ans column number of center of computer frame memory
        self.Cx=imageShape[0]/2
        self.Cy=imageShape[1]/2
        self.C=[self.Cx, self.Cy]

class Data():
    def __init__(self, pointsPath,params):

        self.computerImage=None
        self.realImage=None
        self.computerCentered=None
        self.world=None
        self.n=None

        self.loadData(pointsPath)
        self.calculateImage(params)

    def loadData(self, pointsPath):
        data = np.loadtxt(pointsPath)
        self.n=data.shape[0]
        points3d = data[:, :3]
        points2d = data[:, 3:5] #camera
        # points2d = data[:, 5:] #projecteur
        xw=points3d[:,0]; yw=points3d[:,1]; zw=points3d[:,2];
        Xf=points2d[:,0]; Yf=points2d[:,1]
        self.computerImage=[Xf, Yf]
        self.world = [xw, yw, zw]

    def calculateImage(self, params):
        Xf, Yf = self.computerImage

        Xd=params.dxx*(Xf-params.Cx)
        Yd=params.dy*(Yf-params.Cy)
        self.realImage=[Xd, Yd]

        X=Xf-params.Cx
        Y=Yf-params.Cy
        self.computerCentered=[X,Y]


def solve7unknowns(n, realImage, world):
    xw, yw, zw = world
    Xd, Yd = realImage

    A = np.zeros((n,7))
    for i in range(n):
        A[i,:]=np.array([ Yd[i]*xw[i], Yd[i]*yw[i], Yd[i]*zw[i], Yd[i], -Xd[i]*xw[i], -Xd[i]*yw[i], -Xd[i]*zw[i] ]  )
    # Solve Ax=b avec décomposition QR (b=Xd)
    Q,R = np.linalg.qr(A) # qr decomposition of A
    QTb = np.dot(Q.T,Xd) # computing Q^T*b (project b onto the range of A)
    sol= np.linalg.solve(R,QTb) # solving R*x = Q^T*b
    return sol

def matchingSigns(x, y):
    return x*y>=0

def calculateTy(L, computerCentered,  world):
    X, Y = computerCentered
    xw, yw, zw = world

    #find the point that's furthest from the center of the image
    i=np.argmax( X**2 + Y**2 )
    xwi, ywi, zwi = xw[i],yw[i],zw[i]
    Xi, Yi= X[i], Y[i]

    Ty = 1.0 / np.sqrt( (L[4]*L[4]) + (L[5]*L[5]) + (L[6]*L[6]) )
    #the above calculation gives the absolute value of Ty

    #to find the correct sign we assume Ty is positive...
    r1 = L[0]*Ty ; r2=L[1]*Ty; r3=L[2]*Ty;
    r4 = L[4]*Ty ; r5 = L[5]*Ty ; r6 = L[6]*Ty
    Tx=L[3]*Ty
    x = r1*xwi + r2*ywi + r3*zwi + Tx; y = r4*xwi + r5*ywi +r6*zwi + Ty

    #check our assumption
    if (matchingSigns(x, Xi) and matchingSigns(y, Yi)):
        return Ty
    else:
        return -Ty

def calculateSx(L, Ty):
    return abs(Ty) * np.sqrt( (L[0]*L[0]) + (L[1]*L[1]) + (L[2]*L[2]) )


def calculateRotation(L, Ty, sx):
    # The first two rows can be calculated from L, Ty and sx
    r1 = np.array([ L[0], L[1], L[2] ], np.float64) * (Ty / sx)
    r2 = np.array([ L[4], L[5], L[6] ], np.float64) * Ty
    # because r is orthonormal, row 3 can be calculated from row 1 and row 2
    r3 = np.cross(r1, r2)
    return np.hstack((r1, r2, r3)).reshape(3,3)
def matrixToEuler(M):
    heading = atan2(-M[2][0], M[0][0])
    attitude = asin(M[1][0])
    bank = atan2(-M[1][2], M[1][1])
    angles = [heading, attitude, bank]
    return angles
def eulerToMatrix(angles):
    heading, attitude, bank = angles
    # Convert euler angles back to matrix
    sa, ca = sin(attitude), cos(attitude)
    sb, cb = sin(bank), cos(bank)
    sh, ch = sin(heading), cos(heading)
    return np.array([ [ch*ca, (-ch*sa*cb) + (sh*sb), (ch*sa*sb) + (sh*cb)],
        [sa, ca*cb,-ca*sb],
        [-sh*ca, (sh*sa*cb) + (ch*sb), (-sh*sa*sb) + (ch*cb)] ])
def makeOrthonormal(M):
    return eulerToMatrix(matrixToEuler(M))



def calculateTx(L, Ty, sx):
    return L[3] * Ty / sx


def approximateFTz(R, Ty, world, computerCentered, n, dy):
    X,Y = computerCentered

    arr = R@world
    y=arr[1]+Ty; w=arr[2]

    A = np.column_stack([y, - dy*Y])
    b = w*dy*Y

    sol=np.linalg.lstsq(A,b, rcond=None)
    return sol


def calibrate(pointsPath, imageShape, pixelSize):
    params=Params(imageShape, pixelSize)
    data = Data(pointsPath,params)
    L = solve7unknowns(data.n, data.realImage, data.world)
    Ty = calculateTy(L, data.computerCentered, data.world)
    sx = calculateSx(L, Ty)
    R_test = calculateRotation(L,Ty,sx)
    R = makeOrthonormal(R_test)
    Tx = calculateTx(L,Ty, sx)
    sol = approximateFTz(R, Ty, data.world, data.computerCentered, data.n, params.dy)
    f, Tz = sol[0][0], sol[0][1]
    T = np.array([Tx, Ty, Tz])
    return data, params, R,T, f, sx















#
