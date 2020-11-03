# Check if build/ dir exists.
cd /Users/mariannelado-roy/camera-projector-calibration/unwrapping

python3 channelFix.py

cl3ds_match -m phaseshift_matcher.xml -k "cam match" -o cam_match.png
