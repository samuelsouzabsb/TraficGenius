"""
02_compute_h3_features.py
=========================
Lê os parquets intermediários (nodes, ways, landuse) de uma região e
agrega as features por célula H3 nas resoluções 9, 10 e 11.

Usa interpolação ao longo das ways para distribuir corretamente comprimento
e curvatura entre múltiplas células H3.

Uso:
    python 02_compute_h3_features.py --region NOME

Saídas (em dataset/infra/h3_features/{h3_9,h3_10,h3_11}/):
    h3_9_{region}.parquet
    h3_10_{region}.parquet
    h3_11_{region}.parquet
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import h3
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR  = Path(__file__).resolve().parents[2]
PROC_DIR  = BASE_DIR / "dataset" / "infra" / "processed"
FEAT_DIR  = BASE_DIR / "dataset" / "infra" / "h3_features"

(FEAT_DIR / "h3_9").mkdir(parents=True, exist_ok=True)
(FEAT_DIR / "h3_10").mkdir(parents=True, exist_ok=True)
(FEAT_DIR / "h3_11").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
RESOLUTIONS = [9, 10, 11]
INTERPOLATION_STEP_M = 50.0  # metros entre pontos interpolados nas ways

PLACE_RANK = {
    "city": 4, "town": 3, "suburb": 3,
    "village": 2, "neighbourhood": 2,
    "hamlet": 1,
}

# Hierarquia de tipos de via para calcular "dominante"
HIGHWAY_RANK = {
    "motorway": 10, "motorway_link": 9,
    "trunk": 8,     "trunk_link": 7,
    "primary": 6,   "primary_link": 5,
    "secondary": 4, "secondary_link": 3,
    "tertiary": 2,  "tertiary_link": 2,
    "unclassified": 1, "residential": 1,
    "living_street": 0, "service": 0,
    "track": 0, "path": 0, "road": 0,
    "pedestrian": 0, "cycleway": 0,
}

# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2)**2
    return 2 * R * math.asin(math.sqrt(min(1.0, a)))


def interpolate_line(coords, step_m=INTERPOLATION_STEP_M):
    """
    Interpola pontos ao longo de uma polilinha a cada step_m metros.
    Retorna lista de (lat, lon, frac_length) onde frac_length é a fração
    do comprimento total que cada segmento representa.
    """
    if len(coords) < 2:
        return [(coords[0][0], coords[0][1], 1.0)] if coords else []

    result = []
    total_len = sum(
        haversine_m(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1])
        for i in range(len(coords) - 1)
    )
    if total_len == 0:
        return [(coords[0][0], coords[0][1], 1.0)]

    for i in range(len(coords) - 1):
        p0 = coords[i]
        p1 = coords[i + 1]
        seg_len = haversine_m(p0[0], p0[1], p1[0], p1[1])
        if seg_len == 0:
            continue

        n_steps = max(1, int(seg_len / step_m))
        for k in range(n_steps):
            t = k / n_steps
            lat = p0[0] + t * (p1[0] - p0[0])
            lon = p0[1] + t * (p1[1] - p0[1])
            frac = (seg_len / n_steps) / total_len
            result.append((lat, lon, frac))

    # Último ponto
    result.append((coords[-1][0], coords[-1][1], 0.0))
    return result


def latlng_to_cells(lat, lon):
    """Retorna (h3_9, h3_10, h3_11) para um ponto."""
    return (
        h3.latlng_to_cell(lat, lon, 9),
        h3.latlng_to_cell(lat, lon, 10),
        h3.latlng_to_cell(lat, lon, 11),
    )


def safe_mode(series):
    """Moda segura — retorna o valor mais frequente ou None."""
    if series.empty:
        return None
    return series.mode().iloc[0]


def dominant_highway(series):
    """Retorna o tipo de via de maior hierarquia presente na série."""
    if series.empty:
        return None
    ranked = series.map(lambda x: (HIGHWAY_RANK.get(x, -1), x))
    return max(ranked, key=lambda x: x[0])[1]


# ---------------------------------------------------------------------------
# Agregação de NODES (cruzamentos, semáforos, POIs, places)
# ---------------------------------------------------------------------------

def aggregate_nodes(df_nodes: pd.DataFrame) -> dict:
    """
    Atribui células H3 e agrega contagens de cada tipo de nó.
    Retorna dict com DataFrames para cada resolução.
    """
    if df_nodes.empty:
        return {9: pd.DataFrame(), 10: pd.DataFrame(), 11: pd.DataFrame()}

    print("  -> Atribuindo células H3 aos nós (vetorizado) ...")
    df_nodes = df_nodes.copy()
    df_nodes["h3_9"]  = [h3.latlng_to_cell(lat, lon, 9) for lat, lon in zip(df_nodes.lat, df_nodes.lon)]
    df_nodes["h3_10"] = [h3.latlng_to_cell(lat, lon, 10) for lat, lon in zip(df_nodes.lat, df_nodes.lon)]
    df_nodes["h3_11"] = [h3.latlng_to_cell(lat, lon, 11) for lat, lon in zip(df_nodes.lat, df_nodes.lon)]

    results = {}
    
    # H3-11
    df_nodes_11 = df_nodes[df_nodes["feature_type"].isin(["cruzamento", "semaforo"])]
    if not df_nodes_11.empty:
        nodes_agg_11 = df_nodes_11.groupby(["h3_11", "feature_type"]).size().unstack(fill_value=0)
        nodes_agg_11 = nodes_agg_11.rename(columns={"cruzamento": "n_cruzamentos", "semaforo": "n_semaforos"}).reset_index()
    else:
        nodes_agg_11 = pd.DataFrame(columns=["h3_11", "n_cruzamentos", "n_semaforos"])
    results[11] = nodes_agg_11

    # H3-10
    df_nodes_10 = df_nodes[df_nodes["feature_type"].isin(["posto", "restaurante", "escola"])]
    if not df_nodes_10.empty:
        nodes_agg_10 = df_nodes_10.groupby(["h3_10", "feature_type"]).size().unstack(fill_value=0)
        nodes_agg_10 = nodes_agg_10.rename(columns={"posto": "n_postos", "restaurante": "n_restaurantes", "escola": "n_escolas"}).reset_index()
    else:
        nodes_agg_10 = pd.DataFrame(columns=["h3_10", "n_postos", "n_restaurantes", "n_escolas"])
    results[10] = nodes_agg_10

    # H3-9
    df_nodes_9 = df_nodes[df_nodes["feature_type"] == "hospital"]
    if not df_nodes_9.empty:
        nodes_agg_9 = df_nodes_9.groupby(["h3_9", "feature_type"]).size().unstack(fill_value=0)
        nodes_agg_9 = nodes_agg_9.rename(columns={"hospital": "n_hospitais"}).reset_index()
    else:
        nodes_agg_9 = pd.DataFrame(columns=["h3_9", "n_hospitais"])

    # Places no H3-9
    df_places = df_nodes[df_nodes["feature_type"] == "place"].copy()
    if not df_places.empty:
        df_places["place_rank"] = df_places["subtype"].map(PLACE_RANK).fillna(0).astype(int)
        places_agg = df_places.groupby("h3_9").agg(
            place_type=("place_rank", "max"),
            place_count=("place_rank", "count")
        ).reset_index()
        nodes_agg_9 = nodes_agg_9.merge(places_agg, on="h3_9", how="outer").fillna(0)
        nodes_agg_9["place_type"] = nodes_agg_9["place_type"].astype(int)
        nodes_agg_9["place_count"] = nodes_agg_9["place_count"].astype(int)
        nodes_agg_9["n_hospitais"] = nodes_agg_9["n_hospitais"].fillna(0).astype(int)
    else:
        nodes_agg_9["place_type"] = 0
        nodes_agg_9["place_count"] = 0
    results[9] = nodes_agg_9

    return results


# ---------------------------------------------------------------------------
# Agregação de WAYS (por interpolação vetorizada)
# ---------------------------------------------------------------------------

def aggregate_ways(df_ways: pd.DataFrame) -> dict:
    """
    Divide as ways em subsegmentos e distribui as features para as células H3
    de forma vetorizada e extremamente rápida.
    """
    if df_ways.empty:
        return {9: pd.DataFrame(), 10: pd.DataFrame(), 11: pd.DataFrame()}

    print(f"  -> Processando e extraindo subsegmentos de {len(df_ways):,} ways ...")

    records = []
    
    # Processamento em lista/numpy para evitar lentidão do dataframe row-by-row
    for row in tqdm(df_ways.itertuples(), total=len(df_ways), leave=False):
        coords = json.loads(row.coords_json)
        if len(coords) < 2:
            continue
            
        coords = np.array(coords)
        lats = coords[:, 0]
        lons = coords[:, 1]
        
        # Calcular distâncias dos segmentos usando Haversine vetorizado
        phi1 = np.radians(lats[:-1])
        phi2 = np.radians(lats[1:])
        dphi = np.radians(lats[1:] - lats[:-1])
        dlam = np.radians(lons[1:] - lons[:-1])
        a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2)**2
        dists = 2 * 6_371_000.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        
        total_len = row.length_m
        if total_len == 0:
            continue
            
        ref_name = row.ref if row.ref else (row.name if row.name else None)
            
        for i in range(len(dists)):
            d = dists[i]
            if d == 0:
                continue
                
            lat0, lon0 = lats[i], lons[i]
            lat1, lon1 = lats[i+1], lons[i+1]
            
            # Se o segmento for curto (<=150m), usamos apenas o ponto médio (rapidez)
            if d <= 150.0:
                mid_lat = (lat0 + lat1) / 2.0
                mid_lon = (lon0 + lon1) / 2.0
                frac = d / total_len
                records.append({
                    "lat": mid_lat, "lon": mid_lon, "length_m": d, "frac": frac,
                    "highway": row.highway, "maxspeed": row.maxspeed, "lanes": row.lanes,
                    "curv_accumulated": row.curv_accumulated, "curv_max_deviation": row.curv_max_deviation,
                    "curv_sharp_count": row.curv_sharp_count, "bridge": row.bridge, "tunnel": row.tunnel,
                    "is_roundabout": row.is_roundabout, "ref_name": ref_name
                })
            else:
                # Segmento longo, interpolar a cada 100m
                n_steps = max(1, int(d / 100.0))
                for k in range(n_steps):
                    t = (k + 0.5) / n_steps
                    sub_lat = lat0 + t * (lat1 - lat0)
                    sub_lon = lon0 + t * (lon1 - lon0)
                    sub_d = d / n_steps
                    frac = sub_d / total_len
                    records.append({
                        "lat": sub_lat, "lon": sub_lon, "length_m": sub_d, "frac": frac,
                        "highway": row.highway, "maxspeed": row.maxspeed, "lanes": row.lanes,
                        "curv_accumulated": row.curv_accumulated, "curv_max_deviation": row.curv_max_deviation,
                        "curv_sharp_count": row.curv_sharp_count, "bridge": row.bridge, "tunnel": row.tunnel,
                        "is_roundabout": row.is_roundabout, "ref_name": ref_name
                    })

    if not records:
        return {9: pd.DataFrame(), 10: pd.DataFrame(), 11: pd.DataFrame()}

    df_segs = pd.DataFrame(records)
    del records
    
    # Otimizar tipos de dados para economizar memória
    df_segs["highway"] = df_segs["highway"].astype("category")
    
    print(f"  -> Calculando células H3 para {len(df_segs):,} subsegmentos ...")
    lats_arr = df_segs["lat"].values
    lons_arr = df_segs["lon"].values

    df_segs["h3_9"]  = [h3.latlng_to_cell(lat, lon, 9) for lat, lon in zip(lats_arr, lons_arr)]
    df_segs["h3_10"] = [h3.latlng_to_cell(lat, lon, 10) for lat, lon in zip(lats_arr, lons_arr)]
    df_segs["h3_11"] = [h3.latlng_to_cell(lat, lon, 11) for lat, lon in zip(lats_arr, lons_arr)]

    # Dropar lat/lon imediatamente pois não são mais necessários
    df_segs.drop(columns=["lat", "lon"], inplace=True)

    # Colunas auxiliares para médias ponderadas, deletando colunas originais assim que calculadas
    df_segs["speed_frac"] = df_segs["maxspeed"] * df_segs["frac"]
    df_segs["speed_frac_weight"] = df_segs["frac"].where(df_segs["maxspeed"].notna(), np.nan)
    df_segs.drop(columns=["maxspeed"], inplace=True)

    df_segs["lanes_frac"] = df_segs["lanes"] * df_segs["frac"]
    df_segs["lanes_frac_weight"] = df_segs["frac"].where(df_segs["lanes"].notna() & (df_segs["lanes"] > 0), np.nan)
    df_segs.drop(columns=["lanes"], inplace=True)

    df_segs["curv_acc_frac"] = df_segs["curv_accumulated"] * df_segs["frac"]
    df_segs.drop(columns=["curv_accumulated"], inplace=True)
    
    df_segs["curv_sharp_frac"] = df_segs["curv_sharp_count"] * df_segs["frac"]
    df_segs.drop(columns=["curv_sharp_count"], inplace=True)

    df_segs["bridge_frac"] = df_segs["frac"].where(df_segs["bridge"], 0.0)
    df_segs["bridge_length"] = df_segs["length_m"].where(df_segs["bridge"], 0.0)
    df_segs.drop(columns=["bridge"], inplace=True)

    df_segs["tunnel_frac"] = df_segs["frac"].where(df_segs["tunnel"], 0.0)
    df_segs["tunnel_length"] = df_segs["length_m"].where(df_segs["tunnel"], 0.0)
    df_segs.drop(columns=["tunnel"], inplace=True)

    df_segs["roundabout_frac"] = df_segs["frac"].where(df_segs["is_roundabout"], 0.0)
    df_segs.drop(columns=["is_roundabout"], inplace=True)
    
    df_segs["highway_rank"] = df_segs["highway"].map(HIGHWAY_RANK).fillna(-1).astype(np.int8)

    results = {}

    # --- H3-11 ---
    print("  -> Agregando H3-11 ...")
    g11 = df_segs.groupby("h3_11")
    df_res_11 = g11.agg(
        road_length_m=("length_m", "sum"),
        speed_sum=("speed_frac", "sum"),
        speed_weight=("speed_frac_weight", "sum"),
        lanes_sum=("lanes_frac", "sum"),
        lanes_weight=("lanes_frac_weight", "sum"),
        curv_accumulated_sum=("curv_acc_frac", "sum"),
        curv_weight=("frac", "sum"),
        curv_max_deviation=("curv_max_deviation", "max"),
        curv_sharp_count=("curv_sharp_frac", "sum"),
        n_pontes=("bridge_frac", "sum"),
        bridge_length_m=("bridge_length", "sum"),
        n_tuneis=("tunnel_frac", "sum"),
        tunnel_length_m=("tunnel_length", "sum"),
        n_rotatorias=("roundabout_frac", "sum"),
    )
    df_res_11["speed_mean"] = df_res_11["speed_sum"] / df_res_11["speed_weight"]
    df_res_11["lanes_mean"] = df_res_11["lanes_sum"] / df_res_11["lanes_weight"]
    df_res_11["curv_accumulated"] = df_res_11["curv_accumulated_sum"] / df_res_11["curv_weight"]
    df_res_11["curv_sharp_count"] = np.round(df_res_11["curv_sharp_count"]).astype(int)
    df_res_11["n_pontes"] = np.round(df_res_11["n_pontes"]).astype(int)
    df_res_11["n_tuneis"] = np.round(df_res_11["n_tuneis"]).astype(int)
    df_res_11["n_rotatorias"] = np.round(df_res_11["n_rotatorias"]).astype(int)
    
    # Dominant highway H3-11
    df_hw_11 = df_segs.groupby(["h3_11", "highway"]).agg(sum_frac=("frac", "sum"), highway_rank=("highway_rank", "first")).reset_index()
    df_hw_sorted_11 = df_hw_11.sort_values(["h3_11", "sum_frac", "highway_rank"])
    df_dom_hw_11 = df_hw_sorted_11.drop_duplicates(subset=["h3_11"], keep="last")[["h3_11", "highway"]].rename(columns={"highway": "dominant_highway"})
    df_res_11 = df_res_11.join(df_dom_hw_11.set_index("h3_11")).reset_index()
    results[11] = df_res_11

    # --- H3-10 ---
    print("  -> Agregando H3-10 ...")
    g10 = df_segs.groupby("h3_10")
    df_res_10 = g10.agg(road_length_m=("length_m", "sum"))
    
    df_hw_10 = df_segs.groupby(["h3_10", "highway"]).agg(sum_frac=("frac", "sum"), highway_rank=("highway_rank", "first")).reset_index()
    df_hw_sorted_10 = df_hw_10.sort_values(["h3_10", "sum_frac", "highway_rank"])
    df_dom_hw_10 = df_hw_sorted_10.drop_duplicates(subset=["h3_10"], keep="last")[["h3_10", "highway"]].rename(columns={"highway": "dominant_highway"})
    df_res_10 = df_res_10.join(df_dom_hw_10.set_index("h3_10")).reset_index()
    results[10] = df_res_10

    # --- H3-9 ---
    print("  -> Agregando H3-9 ...")
    g9 = df_segs.groupby("h3_9")
    df_res_9 = g9.agg(
        road_length_m=("length_m", "sum"),
        road_count_distinct=("ref_name", "nunique"),
        curv_sharp_count=("curv_sharp_frac", "sum")
    )
    df_res_9["curv_sharp_count"] = np.round(df_res_9["curv_sharp_count"]).astype(int)
    
    df_hw_9 = df_segs.groupby(["h3_9", "highway"]).agg(sum_frac=("frac", "sum"), highway_rank=("highway_rank", "first")).reset_index()
    df_hw_sorted_9 = df_hw_9.sort_values(["h3_9", "sum_frac", "highway_rank"])
    df_dom_hw_9 = df_hw_sorted_9.drop_duplicates(subset=["h3_9"], keep="last")[["h3_9", "highway"]].rename(columns={"highway": "dominant_highway"})
    df_res_9 = df_res_9.join(df_dom_hw_9.set_index("h3_9")).reset_index()
    results[9] = df_res_9

    return results


# ---------------------------------------------------------------------------
# Agregação de LANDUSE (áreas)
# ---------------------------------------------------------------------------

def aggregate_landuse(df_landuse: pd.DataFrame) -> pd.DataFrame:
    """Agrega áreas de landuse por célula H3-9."""
    if df_landuse.empty:
        return pd.DataFrame()

    print("  -> Atribuindo células H3-9 às áreas de landuse (vetorizado) ...")
    df_landuse = df_landuse.copy()
    df_landuse["h3_9"] = [
        h3.latlng_to_cell(lat, lon, 9)
        for lat, lon in zip(df_landuse.centroid_lat, df_landuse.centroid_lon)
    ]

    urban = df_landuse[df_landuse["landuse_class"] == "urban"].groupby("h3_9")["area_m2"].sum()
    rural = df_landuse[df_landuse["landuse_class"] == "rural"].groupby("h3_9")["area_m2"].sum()

    df_lu = pd.DataFrame({"urban_area_m2": urban, "rural_area_m2": rural}).fillna(0).reset_index()
    df_lu["urban_ratio"] = df_lu["urban_area_m2"] / (df_lu["urban_area_m2"] + df_lu["rural_area_m2"] + 1e-9)
    df_lu["urban_ratio"] = df_lu["urban_ratio"].where(
        (df_lu["urban_area_m2"] + df_lu["rural_area_m2"]) > 0, other=None
    )
    return df_lu


# ---------------------------------------------------------------------------
# Agregação de speed_min / speed_max (vetorizada sobre centroid)
# ---------------------------------------------------------------------------

def aggregate_speed_minmax(df_ways: pd.DataFrame) -> dict:
    """
    Para speed_min e speed_max, usa o centroide das ways (mais rápido).
    Resultado adicionado às features H3-11.
    """
    if df_ways.empty:
        return {}

    df_sp = df_ways[df_ways["maxspeed"].notna()].copy()
    if df_sp.empty:
        return {}

    df_sp["h3_11"] = [
        h3.latlng_to_cell(lat, lon, 11)
        for lat, lon in zip(df_sp.centroid_lat, df_sp.centroid_lon)
    ]
    g = df_sp.groupby("h3_11")["maxspeed"]
    return {"speed_min": g.min(), "speed_max": g.max()}


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def compute_h3_features(region: str):
    print(f"\n{'='*60}")
    print(f"  Calculando features H3 para: {region}")
    print(f"{'='*60}")

    out_9  = FEAT_DIR / "h3_9"  / f"h3_9_{region}.parquet"
    out_10 = FEAT_DIR / "h3_10" / f"h3_10_{region}.parquet"
    out_11 = FEAT_DIR / "h3_11" / f"h3_11_{region}.parquet"

    if out_9.exists() and out_10.exists() and out_11.exists():
        print(f"  [SKIP] Features H3 de {region} já existem.")
        return

    # Carregar parquets intermediários
    nodes_path   = PROC_DIR / f"nodes_{region}.parquet"
    ways_path    = PROC_DIR / f"ways_{region}.parquet"
    landuse_path = PROC_DIR / f"landuse_{region}.parquet"

    for p in [nodes_path, ways_path, landuse_path]:
        if not p.exists():
            # Tentar com .osm.parquet caso tenha sido salvo assim
            alt_path = p.with_name(p.name.replace(".parquet", ".osm.parquet"))
            if alt_path.exists():
                if p == nodes_path: nodes_path = alt_path
                elif p == ways_path: ways_path = alt_path
                elif p == landuse_path: landuse_path = alt_path
            else:
                print(f"  ERRO: arquivo não encontrado: {p}")
                print(f"  Execute primeiro: python 01_extract_osm_features.py <pbf> --region {region}")
                return

    print("  Carregando parquets intermediários ...")
    df_nodes   = pd.read_parquet(nodes_path)
    df_ways    = pd.read_parquet(ways_path)
    df_landuse = pd.read_parquet(landuse_path)

    print(f"  Nodes: {len(df_nodes):,} | Ways: {len(df_ways):,} | Landuse: {len(df_landuse):,}")

    # Agregar nodes
    print("\n  [1/4] Agregando nodes por H3 ...")
    node_feats = aggregate_nodes(df_nodes)

    # Agregar ways
    print("\n  [2/4] Agregando ways por H3 (interpolação otimizada) ...")
    way_feats = aggregate_ways(df_ways)

    # Agregar landuse
    print("\n  [3/4] Agregando landuse por H3-9 ...")
    landuse_feats = aggregate_landuse(df_landuse)

    # Speed min/max
    print("\n  [4/4] Calculando speed_min / speed_max ...")
    sp_minmax = aggregate_speed_minmax(df_ways)

    # -----------------------------------------------------------------------
    # Montar DataFrame final por resolução
    # -----------------------------------------------------------------------
    print("\n  Consolidando features por resolução ...")

    # --- H3-11 ---
    base_11 = way_feats.get(11, pd.DataFrame())
    nodes_11 = node_feats.get(11, pd.DataFrame())

    if not base_11.empty and not nodes_11.empty:
        df_11 = base_11.merge(nodes_11, on="h3_11", how="outer")
    elif not base_11.empty:
        df_11 = base_11
    elif not nodes_11.empty:
        df_11 = nodes_11
    else:
        df_11 = pd.DataFrame()

    # Adicionar speed_min / speed_max
    if not df_11.empty and sp_minmax:
        df_11 = df_11.merge(
            pd.DataFrame(sp_minmax).reset_index(),
            on="h3_11", how="left"
        )

    # Colunas H3-11 finais
    cols_11 = [
        "h3_11",
        "n_cruzamentos", "n_semaforos",
        "speed_mean", "speed_min", "speed_max",
        "lanes_mean",
        "curv_accumulated", "curv_max_deviation", "curv_sharp_count",
        "n_rotatorias", "n_pontes", "bridge_length_m",
        "n_tuneis", "tunnel_length_m",
        "road_length_m", "dominant_highway",
    ]
    for c in cols_11:
        if c not in df_11.columns:
            df_11[c] = None
    df_11 = df_11[[c for c in cols_11 if c in df_11.columns]]

    # --- H3-10 ---
    base_10 = way_feats.get(10, pd.DataFrame())
    nodes_10 = node_feats.get(10, pd.DataFrame())

    if not base_10.empty and not nodes_10.empty:
        df_10 = base_10.merge(nodes_10, on="h3_10", how="outer")
    elif not base_10.empty:
        df_10 = base_10
    elif not nodes_10.empty:
        df_10 = nodes_10
    else:
        df_10 = pd.DataFrame()

    cols_10 = [
        "h3_10",
        "n_postos", "n_restaurantes", "n_escolas",
        "road_length_m", "dominant_highway",
    ]
    for c in cols_10:
        if c not in df_10.columns:
            df_10[c] = None
    if not df_10.empty:
        df_10 = df_10[[c for c in cols_10 if c in df_10.columns]]

    # --- H3-9 ---
    base_9 = way_feats.get(9, pd.DataFrame())
    nodes_9 = node_feats.get(9, pd.DataFrame())

    if not base_9.empty and not nodes_9.empty:
        df_9 = base_9.merge(nodes_9, on="h3_9", how="outer")
    elif not base_9.empty:
        df_9 = base_9
    elif not nodes_9.empty:
        df_9 = nodes_9
    else:
        df_9 = pd.DataFrame()

    # Adicionar landuse
    if not df_9.empty and not landuse_feats.empty:
        df_9 = df_9.merge(landuse_feats, on="h3_9", how="left")
    elif not landuse_feats.empty:
        df_9 = landuse_feats

    cols_9 = [
        "h3_9",
        "n_hospitais",
        "road_total_length_m", "road_count_distinct", "total_sharp_curves",
        "place_type", "place_count",
        "dominant_highway_r9",
        "urban_area_m2", "rural_area_m2", "urban_ratio",
    ]
    # Renomear colunas de way_feats para nomes finais de H3-9
    if not df_9.empty:
        rename_map = {}
        if "road_length_m" in df_9.columns:
            rename_map["road_length_m"] = "road_total_length_m"
        if "curv_sharp_count" in df_9.columns:
            rename_map["curv_sharp_count"] = "total_sharp_curves"
        if "dominant_highway" in df_9.columns:
            rename_map["dominant_highway"] = "dominant_highway_r9"
        if rename_map:
            df_9 = df_9.rename(columns=rename_map)

    for c in cols_9:
        if c not in df_9.columns:
            df_9[c] = None
    if not df_9.empty:
        available = [c for c in cols_9 if c in df_9.columns]
        df_9 = df_9[available]

    # -----------------------------------------------------------------------
    # Salvar
    # -----------------------------------------------------------------------
    print("\n  Salvando features H3 ...")

    if not df_11.empty:
        df_11.to_parquet(out_11, index=False)
        print(f"    -> h3_11_{region}.parquet: {len(df_11):,} células")
    else:
        pd.DataFrame().to_parquet(out_11, index=False)

    if not df_10.empty:
        df_10.to_parquet(out_10, index=False)
        print(f"    -> h3_10_{region}.parquet: {len(df_10):,} células")
    else:
        pd.DataFrame().to_parquet(out_10, index=False)

    if not df_9.empty:
        df_9.to_parquet(out_9, index=False)
        print(f"    -> h3_9_{region}.parquet: {len(df_9):,} celulas")
    else:
        pd.DataFrame().to_parquet(out_9, index=False)

    print(f"\n  OK Features H3 de {region} concluidas!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agrega features OSM por celulas H3")
    parser.add_argument("--region", required=True, help="Nome da regiao")
    args = parser.parse_args()
    compute_h3_features(args.region)

