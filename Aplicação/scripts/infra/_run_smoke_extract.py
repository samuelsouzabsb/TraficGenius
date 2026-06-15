import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

# Adicionar diretório ao path
sys.path.insert(0, r'C:\Users\samue\Documents\trafic\scripts\infra')
os.chdir(r'C:\Users\samue\Documents\trafic')

exec(open(r'C:\Users\samue\Documents\trafic\scripts\infra\01_extract_osm_features.py', encoding='utf-8').read())
process_pbf(r'dataset\infra\us\delaware-260612.osm.pbf', 'us_delaware')
