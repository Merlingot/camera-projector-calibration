# Check if build/ dir exists.

# RUN FROM /unwrapping DIRECTORY ONLY

python3 channelFix.py

cl3ds_match -m phaseshift_matcher.xml -k "cam match" -o cam_match.png
