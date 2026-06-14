import pandas as pd
import psutil
arquivo = r"C:\Users\samuelbarroso\Downloads\dados_unificados5.parquet"
import gc
import pandas as pd
import geopandas as gpd
import h3
from pyrosm import OSM

ARQUIVO_OSM = r"C:\Users\samuelbarroso\Downloads\north-america-latest.osm.pbf"

SAIDA = r"C:\Users\samuelbarroso\Downloads\features"

print("Abrindo OSM...")

osm = OSM(ARQUIVO_OSM)

print("Extraindo rodovias...")

roads = osm.get_network(
    network_type="driving"
)

print("Quantidade:", len(roads))

roads = roads.to_crs(4326)

centroid = roads.geometry.centroid

roads["lat"] = centroid.y
roads["lon"] = centroid.x

print("Gerando H3...")

roads["h3_11"] = roads.apply(
    lambda x: h3.latlng_to_cell(
        x["lat"],
        x["lon"],
        11
    ),
    axis=1
)

roads["h3_10"] = roads["h3_11"].apply(
    lambda x: h3.cell_to_parent(
        x,
        10
    )
)

roads["h3_9"] = roads["h3_11"].apply(
    lambda x: h3.cell_to_parent(
        x,
        9
    )
)

roads["lanes_num"] = pd.to_numeric(
    roads["lanes"],
    errors="coerce"
)

roads["maxspeed_num"] = pd.to_numeric(
    roads["maxspeed"]
         .astype(str)
         .str.extract(r"(\d+)")[0],
    errors="coerce"
)

roads["bridge_flag"] = (
    roads["bridge"]
    .fillna("no")
    .astype(str)
    .str.lower()
    .eq("yes")
    .astype(int)
)

roads["tunnel_flag"] = (
    roads["tunnel"]
    .fillna("no")
    .astype(str)
    .str.lower()
    .eq("yes")
    .astype(int)
)

roads["roundabout_flag"] = (
    roads["junction"]
    .fillna("")
    .astype(str)
    .str.lower()
    .eq("roundabout")
    .astype(int)
)

roads_proj = roads.to_crs(3857)

roads["road_length_m"] = (
    roads_proj.geometry.length
)

colunas = [
    "h3_9",
    "h3_10",
    "h3_11",
    "lanes_num",
    "maxspeed_num",
    "bridge_flag",
    "tunnel_flag",
    "roundabout_flag",
    "road_length_m"
]

roads = roads[colunas]

arquivo_saida = (
    rf"{SAIDA}\roads_h3.parquet"
)

roads.to_parquet(
    arquivo_saida,
    index=False
)

print("Salvo:", arquivo_saida)

del roads
gc.collect()
