""" Tsai calibration stage 2 : Optimisation non linéaire pour les paramètres f Tz kappa1 kappa2 """

import scipy.optimize as sci
import numpy as np


def formatage0(world, realImage,  R, T, f0, sx):
    Tx, Ty, Tz0 = T
    r1, r2, r3 = R[0,:]
    r4, r5, r6 = R[1,:]
    r7, r8, r9 = R[2,:]
    xw, yw, zw = world
    Xd, Yd = realImage
    args = (r1, r2, r3, r4, r5, r6, r7, r8, r9, xw, yw, zw, Xd, Yd, Tx, Ty)
    theta0 = (f0, Tz0, 0, 0)
    return theta0, args

def costfunction(theta, *args):
    f, Tz, k1, k2 = theta
    r1, r2, r3, r4, r5, r6, r7, r8, r9, xw, yw, zw, Xd, Yd, Tx, Ty = args
    rsquared = Xd**2 + Yd**2
    Fx = f*(r1*xw+r2*yw+r3*zw+Tx)/(r7*xw+r8*yw+r9*zw+Tz) - Xd*(1+ (k1*rsquared + k2*rsquared**2) )
    Fy = f*(r4*xw+r5*yw+r6*zw+Ty)/(r7*xw+r8*yw+r9*zw+Tz) - Yd*(1 + (k1*rsquared +k2*rsquared**2) )
    return np.sum( Fx**2 ) + np.sum( Fy**2 )

def nonLinearSearch(world, realImage, R, T, f, sx):
    theta0, arg = formatage0(world, realImage, R, T, f, sx)
    sol = sci.minimize(costfunction, theta0, args=arg, tol=1e-6)
    print(sol.message)
    print(sol.nit)
    f_, Tz_, k1_, k2_ = sol.x[0], sol.x[1], sol.x[2], sol.x[3]
    return f_, Tz_, k1_, k2_
