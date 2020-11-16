import numpy as np
import cv2

# print("hey")

# img = cv2.imread("circlesgrid.png")
#
# pattern_shape = (4, 11) #patternSize: number of circles per row and column ( patternSize = Size(points_per_row, points_per_colum) ).
#
# gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, corners = cv2.findCirclesGrid(gray_frame, pattern_shape, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
#
# #print(corners)
# #print(corners[0][0])
# drawn_frame = cv2.drawChessboardCorners(img, pattern_shape, corners, ret)
# cv2.circle(drawn_frame, (corners[0][0][0], corners[0][0][1]), 10, (255,255,255), 2)
#
# cv2.imwrite("detect.png", drawn_frame)

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

# asdf = getAsymCirclesObjPoints(11, 4, 1, 0.0, 0.5, "xy")
# print(asdf)
