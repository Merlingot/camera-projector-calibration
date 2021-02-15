import cv2
import numpy as np

class sgmf():
    def __init__(self, sgmfName, projSize, shadowMaskName=None):

        """
        Args:
            sgmfName :  Filename
            projSize :  tuple(x, y)
                        Projector's resolution
            shadowMask : Filename
        """

        sgmfBGR = cv2.imread(sgmfName,cv2.IMREAD_UNCHANGED)/65535
        self.channelX  = sgmfBGR[:,:,2] * projSize[0] # Red channel => X
        self.channelY = sgmfBGR[:,:,1] * projSize[1] # Green channel => Y

        self.shadowMask=np.ones(sgmfBGR.shape[0:1])
        if shadowMaskName:
            self.shadowMask = cv2.imread(shadowMaskName)

    def get_value(self, vecPix):
        u,v=vecPix[0],vecPix[1]

        u_down, v_down = int(u), int(v) #entier
        u_up, v_up = int(u+1), int(v+1) #reste

        f_u_up, f_v_up = u%1, v%1 #fraction reste
        f_u_down, f_v_down = (1-u%1), (1-v%1) #fraction entier

        # INDEXATION LIGNE (v), COLONNE (u) !!!!!!
        # variation en x
        ex_down = self.channelX[v_down,u_down]
        ex_up = self.channelX[v_down,u_up]
        # variation en y
        ey_down = self.channelY[v_down,u_down]
        ey_up = self.channelY[v_up,u_down]

        ex = f_u_down*ex_down + f_u_up*ex_up
        ey = f_v_down*ey_down + f_v_up*ey_up

        return np.array([ex,ey])
