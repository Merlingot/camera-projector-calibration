import numpy as np
import math
import sys
import cv2


params = cv2.structured_light_SinusoidalPattern_Params()
params.width = 1+5*32 #pixel width
params.height = 1+5*32 #pixel height
params.nbrOfPeriods = 5 # amount of waves
params.setMarkers = False # patterns /w or /wo markers
params.horizontal = False
params.methodId = cv2.structured_light.PSP
params.shiftValue = 2.0 * math.pi / 3.0
params.nbrOfPixelsBetweenMarkers = 70

sinus = cv2.structured_light_SinusoidalPattern.create(params)
r, patterns = sinus.generate()
wrapped_phase_map, shadow_mask = sinus.computePhaseMap(patterns)

cam_size = (params.width, params.height)
print("before")
unwrapped_phase_map = wrapped_phase_map.copy()
sinus.unwrapPhaseMap(wrapped_phase_map, cam_size, unwrapped_phase_map, shadow_mask)
print("after")
