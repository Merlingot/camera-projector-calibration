Phase unwrapping:
cl3ds_create_generator_config -m phaseshift
cl3ds_generate -g phaseshift_generator.xml -o fringes/phaseshift_%03d.png

Mesures:
changer SERIE_NO dans ./unwrapping/scan.py
python3 ./scan.py
déplacer ./data/series_<SERIE_NO>/scan_3channels -> ./unwrapping/scan_3channels
sh unwrap.sh
déplacer ./unwrapping/scan_3channels vers ./data/series_<SERIE_NO>/scan_3channels
déplacer ./unwrapping/cam_match.png vers ./data/series_<SERIE_NO>
changer SERIE_NO dans ./confidenceMap.py
python3 ./confidenceMap.py
