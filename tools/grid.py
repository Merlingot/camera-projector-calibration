""" Fonction pour générer les coordonées des centres des cercles dans le référentiel monde"""

import numpy as np
import cv2


def getCorners(rows, cols, spacing, xoffset, yoffset, flag="2d", zvalue=0):
    pts = []
    if flag == "yz":
        xstart = 0
        for y in range(0, rows):
            ypos = y * spacing + yoffset
            for x in range(0, cols):
                xpos = xstart + (x * spacing)
                xpos += xoffset
                pts.append([zvalue, ypos, xpos])
    else:
        xstart = (cols - 1.0) * spacing
        # print("xstart", xstart)
        for y in range(0, rows):
            ypos = y * spacing + yoffset
            for x in range(0, cols):
                xpos = xstart - (x * spacing)
                xpos += xoffset
                if flag == "2d":
                    pts.append([xpos, ypos])
                elif flag == "xy":
                    pts.append([xpos, ypos, zvalue])
                else:  # xz
                    pts.append([xpos, zvalue, ypos])

    return np.array(pts)


def getAsymCirclesObjPoints(rows, cols, spacing, xoffset, yoffset, flag="2d", zvalue=0):
    pts = []
    if flag == "yz":
        xstart = 0
        for y in range(0, rows):
            ypos = y * spacing + yoffset
            for x in range(0, cols):
                xpos = xstart + (2.0 * x * spacing)
                if y % 2 == 1: ## odd rows are shifted 1 spacing to the left
                    xpos += spacing
                xpos += xoffset
                pts.append([zvalue, ypos, xpos])
    else:
        xstart = (2.0 * cols - 1.0) * spacing
        # print("xstart", xstart)
        for y in range(0, rows):
            ypos = y * spacing + yoffset
            for x in range(0, cols):
                xpos = xstart - (2.0 * x * spacing)
                if y % 2 == 1: ## odd rows are shifted 1 spacing to the left
                    xpos -= spacing
                xpos += xoffset
                if flag == "2d":
                    pts.append([xpos, ypos])
                elif flag == "xy":
                    pts.append([xpos, ypos, zvalue])
                else:  # xz
                    pts.append([xpos, zvalue, ypos])

    return np.array(pts)
