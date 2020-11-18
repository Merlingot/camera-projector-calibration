# phaseshift_generator et phaseshift_matcher

Faire un phaseshift_generator pour chaque projecteur
-> cl3ds_create_generator_config -m phaseshift
-> Générer les franges
    cd ./unwrapping
    rm -r fringes/* ou mkdir fringes
    cl3ds_generate -g phaseshift_generator.xml -o fringes/phaseshift_%03d.png
-> Créer un phaseshift_matcher:
cl3ds_create_matcher_config -m phaseshift -c seqreader -p dummy
-> Mettre la résolution du projecteur dans scan.py


# Mesures:

1.  cd unwrapping
2.  Dans ./unwrapping/scan.py -> setter les paramètres de caméra et de résolution de projecteur
3.  changer FOLDER/SERIE dans ./unwrapping/scan.py
4.  python3 ./scan.py

# Traitement des données:

5. Mettre les données de scan_3channels dans le meme folder que le phaseshift_matcher:
-> python3 channelFix.py (pour les images couleurs seulement)
-> cp ./data/serie/scan_3channels ./unwrapping/projecteur/scan
7.  cl3ds_match -m phaseshift_matcher.xml -k "cam match" -o cam_match.png


# Confidence Map
changer SERIE dans ./confidenceMap.py
python3 ./confidenceMap.py
