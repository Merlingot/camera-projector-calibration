import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob, os

# Data:
SERIE="03_12_2020"
dataPath="data/{}/".format(SERIE)
outputPath=os.path.join(dataPath,"output/")
# cam (1)
s = cv.FileStorage()
s.open('{}cam.xml'.format(outputPath), cv.FileStorage_READ)
K1=s.getNode('K').mat()
_=s.getNode('R').mat()
_=s.getNode('t').mat()
D1=s.getNode('coeffs').mat()
imageSize=s.getNode('imageSize').mat()
imageSize=(int(imageSize[0][0]),int(imageSize[1][0]))
s.release()
# proj (2)
s = cv.FileStorage()
s.open('{}proj.xml'.format(outputPath), cv.FileStorage_READ)
K2=s.getNode('K').mat()
R=s.getNode('R').mat()
T=s.getNode('t').mat()
D2=s.getNode('coeffs').mat()
s.release()
# Frame 1 (cam)
r=os.path.join(dataPath,"max_17.png")
img1=cv.imread(r)
img1=cv.normalize(img1.astype(np.uint8), None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX)
# for i in range(20, img1.shape[0], 200):
#     cv.line(img1, (0, i), (img1.shape[1], i), (1, 0, 0), 10)
# Frame 2 (proj)
l=os.path.join(dataPath,"match_17.png")
sgmf=cv.imread(l,cv.IMREAD_UNCHANGED)/65535
sgmf[:,:,2] *= imageSize2[0] # Red channel => X
sgmf[:,:,1] *= imageSize2[1] # Green channel => Y
sgmf[:,:,0] *=0
img2=cv.normalize(sgmf.astype(np.uint8), None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX)
# for i in range(20, img2.shape[0], 200):
#     cv.line(img2, (0, i), (img2.shape[1], i), (1, 0, 0), 10)

# plt.figure()
# plt.imshow(img1)
# plt.figure()
# plt.imshow(img2)
# plt.show()


# Rectification
R1, R2, P1, P2, Q, roi_1, roi_2= cv.stereoRectify(K1, D1, K2, D2, imageSize, R, T, flags=cv.CALIB_ZERO_DISPARITY, alpha=-1)
# cam (1)
MapX1, MapY1 = cv.initUndistortRectifyMap(K1, D1, R1, P1, imageSize, cv.CV_8UC1)

rectified1= cv.remap(img1, MapX1, MapY1, cv.INTER_LINEAR, cv.BORDER_CONSTANT)
# projecteur (2)
MapX2, MapY2 = cv.initUndistortRectifyMap(K2, D2, R2, P2, imageSize, cv.CV_8UC1)
rectified2= cv.remap(img2, MapX2, MapY2, cv.INTER_LINEAR, cv.BORDER_CONSTANT)
# plt.figure()
# plt.imshow(rectified1)
# plt.figure()
# plt.imshow(rectified2)
# plt.show()


img_rect1=rectified1;img_rect2=rectified2;
# draw the images side by side
total_size = (max(img_rect1.shape[0], img_rect2.shape[0]),
              img_rect1.shape[1] + img_rect2.shape[1], 3)
img = np.zeros(total_size)
img[:img_rect1.shape[0], :img_rect1.shape[1]] = img_rect1
img[:img_rect2.shape[0], img_rect1.shape[1]:] = img_rect2

# Stereo mathing
rectified1_= cv.resize(rectified1, None, None, 0.5, 0.5, cv.INTER_LINEAR_EXACT)
rectified2_ = cv.resize(rectified2, None, None, 0.5, 0.5, cv.INTER_LINEAR_EXACT)
for_matcher1 = cv.cvtColor(rectified1_,  cv.COLOR_BGR2GRAY);
for_matcher2 =cv.cvtColor(rectified2_, cv.COLOR_BGR2GRAY);


wsize = 7
max_disp=16
matcher1 = cv.StereoBM_create(max_disp,wsize)
wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher1)
matcher2 = cv.ximgproc.createRightMatcher(matcher1)


disp1=left_matcher.compute(for_matcher1,for_matcher2);
disp2=right_matcher.compute(for_matcher2,for_matcher1);

wls_filter.setLambda(lambda_);
wls_filter.setSigmaColor(sigma);
wls_filter.filter(disp1,left,filtered_disp,right_disp);

help(wls_filter.filter)



#
