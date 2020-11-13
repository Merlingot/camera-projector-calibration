Phase unwrapping:
Faire un phaseshift_generator pour chaque projecteur !
-> changer la résolution dans phaseshift_generator.xml
cd ./unwrapping
rm -r fringes/*
cl3ds_generate -g phaseshift_generator.xml -o fringes/phaseshift_%03d.png
->Changer le nombre de fringes dans phaseshift_matcher

Mesures:
Dans ./unwrapping/scan.py -> setter les paramètres de caméra et de résolution de projecteur
changer SERIE dans ./unwrapping/scan.py
python3 ./scan.py
déplacer ./data/series_<SERIE_NO>/scan_3channels -> ./unwrapping/scan_3channels
sh unwrap.sh
déplacer ./unwrapping/scan_3channels vers ./data/series_<SERIE_NO>/scan_3channels
déplacer ./unwrapping/cam_match.png vers ./data/series_<SERIE_NO>
changer SERIE dans ./confidenceMap.py
python3 ./confidenceMap.py
