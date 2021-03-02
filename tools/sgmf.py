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

        sgmfBGR = cv2.imread(sgmfName,cv2.IMREAD_UNCHANGED)/65535.0
        self.channelX  = sgmfBGR[:,:,2] * (projSize[0] - 1)# Red channel => X
        self.channelY = sgmfBGR[:,:,1] * (projSize[1] - 1) # Green channel => Y

        camWidth = sgmfBGR.shape[1]
        camHeight = sgmfBGR.shape[0]
        self.camSize = (camWidth, camHeight)
        self.projSize = projSize


        self.shadowMask=np.ones(sgmfBGR.shape[0:1])
        if shadowMaskName:
            self.shadowMask = cv2.imread(shadowMaskName)

    def get_value(self, vecPix):
        u,v = int(np.round(vecPix[0])), int(np.round(vecPix[1]))
        # INDEXATION LIGNE (v), COLONNE (u) !!!!!!
        ex, ey = self.channelX[v,u], self.channelY[v,u] #les channels
        return np.array([ex,ey])
