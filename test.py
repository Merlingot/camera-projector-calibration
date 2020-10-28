#!/usr/local/bin/python3

import numpy as np
from cv2 import *
import os
import sys
import matplotlib.pyplot as plt

# Camera et moniteurs ----------------
CAM_OSNUMBER=1
# Dimensions du first monitor
monitor_width = int(1440); monitor_heigth = int(900);
# Dimensions du first monitor
projector_width = int(800) ; projector_heigth = int(600);


# Sinusoidal Patterns------------------------------------
# Set patterns parameters
params = structured_light_SinusoidalPattern_Params()
params.width=projector_width
params.height=projector_heigth
params.nbrOfPeriods=int(3);
params.setMarkers = True
params.horizontal = True
params.methodId = structured_light.PSP
params.shiftValue = 2.0 * np.pi / 3.0
params.nbrOfPixelsBetweenMarkers = 70
# Generate sinusoidal patterns
sinus=structured_light.SinusoidalPattern_create(params)
_, patternImages = sinus.generate()

# Test les window pour le projecteur ----------------------
test_pattern = patternImages[0]
height, width = test_pattern.shape[:2]

namedWindow("pattern", WINDOW_NORMAL);
moveWindow("pattern", 0, 0);
moveWindow("pattern", 0, -monitor_heigth);
imshow("pattern", test_pattern);
waitKey(0)
setWindowProperty("pattern", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
waitKey(0)
destroyAllWindows()
