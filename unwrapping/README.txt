# phaseshift_generator et phaseshift_matcher

Faire un phaseshift_generator pour chaque projecteur.
-> Dans phaseshift_generator.xml : mettre la résolution du projecteur voulu
-> Générer les franges
    cd ./unwrapping
    rm -r fringes/*
    cl3ds_generate -g phaseshift_generator.xml -o fringes/phaseshift_%03d.png
-> Mettre le nombre de fringes dans phaseshift_matcher
-> Mettre la résolution du projecteur dans scan.py


# Mesures:

1.  cd unwrapping
2.  Dans ./unwrapping/scan.py -> setter les paramètres de caméra et de résolution de projecteur
3.  changer SERIE dans ./unwrapping/scan.py
4.  python3 ./scan.py
5.  déplacer ./data/<SERIE>/scan_3channels -> ./unwrapping/scan_3channels
6.  sh unwrap.sh
7.  déplacer ./unwrapping/scan_3channels vers ./data/<SERIE>/scan_3channels
8.  déplacer ./unwrapping/cam_match.png vers ./data/<SERIE>


# Confidence Map

changer SERIE dans ./confidenceMap.py
python3 ./confidenceMap.py
