import numpy as np
import math
import sys
import cv2

cam_size = [-1,-1]

params = cv2.structured_light_SinusoidalPattern_Params()
params.width = 1024 #pixel width
params.height = 1024 #pixel height
params.nbrOfPeriods = 5 # amount of waves
params.setMarkers = False # patterns /w or /wo markers
params.horizontal = False
params.methodId = cv2.structured_light.PSP
params.shiftValue = 2.0 * math.pi / 3.0
params.nbrOfPixelsBetweenMarkers = 70

sinus = cv2.structured_light_SinusoidalPattern.create(params)
r, patterns = sinus.generate()
count = 0



img = []
while count < len(patterns) :
    img.append(patterns[count])
    count += 1

captures = []
captures.append(img[0])
captures.append(img[1])
captures.append(img[2])

wrapped_phase_map, shadow_mask = sinus.computePhaseMap(patterns)
cam_size = (1024, 1024)
print("before")
unwrapped_phase_map = sinus.unwrapPhaseMap(wrapped_phase_map, cam_size, shadow_mask)
print("after")

import matplotlib.pyplot as plt
plt.plot(wrapped_phase_map)
