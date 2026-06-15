"""
01_extract_osm_features.py  (v2 - osmium 4.x FileProcessor API)
================================================================
Extrai features de infraestrutura de um arquivo .osm.pbf em 4 passagens:

  Passagem 1 - NodeDegreeCounter : conta quantas ways referenciam cada no
  Passagem 2 - NodeExtractor     : extrai nos de interesse (cruzamentos, POIs, places)
  Passagem 3 - WayExtractor      : extrai ways com geometria e metricas de curvatura
  Passagem 4 - AreaExtractor     : extrai poligonos de landuse

Uso:
    python 01_extract_osm_features.py <arquivo.pbf> [--region NOME]

Saidas (em dataset/infra/processed/):
    nodes_{region}.parquet
    ways_{region}.parquet
    landuse_{region}.parquet
"""

import sys
import os
import argparse
import math
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import osmium
import osmium.index

# ---------------------------------------------------------------------------
# Configuracoes de paths
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parents[2]
PROC_DIR   = BASE_DIR / "dataset" / "infra" / "processed"
TMP_IDX    = BASE_DIR / "dataset" / "infra" / "tmp_idx"
PROC_DIR.mkdir(parents=True, exist_ok=True)
TMP_IDX.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Tags de interesse
# ---------------------------------------------------------------------------
HIGHWAY_VALUES = {
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "unclassified", "residential",
    "living_street", "service", "track", "path", "road",
    "pedestrian", "cycleway",
}

AMENITY_NODES = {
    "fuel":        "posto",
    "restaurant":  "restaurante",
    "fast_food":   "restaurante",
    "cafe":        "restaurante",
    "school":      "escola",
    "university":  "escola",
    "college":     "escola",
    "hospital":    "hospital",
    "clinic":      "hospital",
    "doctors":     "hospital",
}

PLACE_VALUES = {"city", "town", "village", "hamlet", "suburb", "neighbourhood"}

PLACE_RANK = {
    "city": 4, "town": 3, "suburb": 3,
    "village": 2, "neighbourhood": 2,
    "hamlet": 1,
}

LANDUSE_URBAN = {"residential", "commercial", "industrial", "retail", "construction"}
LANDUSE_RURAL = {"farmland", "forest", "meadow", "pasture", "orchard", "vineyard", "scrub"}

# ---------------------------------------------------------------------------
# Funcoes geometricas auxiliares
# ---------------------------------------------------------------------------

def haversine_m(lat1, lon1, lat2, lon2):
    """Distancia haversine em metros entre dois pontos."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(min(1.0, a)))


def angle_between_segments(p0, p1, p2):
    """Angulo (graus) entre p0->p1 e p1->p2."""
    def vec(a, b):
        return (b[0] - a[0], b[1] - a[1])
    v1 = vec(p0, p1)
    v2 = vec(p1, p2)
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos_a = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.degrees(math.acos(cos_a))


def compute_curvature(coords):
    """
    3 metricas de curvatura a partir de lista de (lat, lon).
    Returns: (curv_accumulated, curv_max_deviation, curv_sharp_count)
    """
    if len(coords) < 3:
        return 0.0, 0.0, 0

    total_len = sum(
        haversine_m(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1])
        for i in range(len(coords) - 1)
    )
    if total_len == 0:
        return 0.0, 0.0, 0

    angles = [
        angle_between_segments(coords[i-1], coords[i], coords[i+1])
        for i in range(1, len(coords) - 1)
    ]
    curv_accumulated = sum(angles) / total_len
    curv_sharp_count = sum(1 for a in angles if a > 30.0)

    lat0, lon0 = coords[0]
    lat1, lon1 = coords[-1]
    line_len = haversine_m(lat0, lon0, lat1, lon1)

    if line_len < 1.0:
        curv_max_deviation = total_len
    else:
        dx = lon1 - lon0
        dy = lat1 - lat0
        max_dev = 0.0
        for lat, lon in coords[1:-1]:
            t = ((lon - lon0) * dx + (lat - lat0) * dy) / (dx*dx + dy*dy + 1e-15)
            t = max(0.0, min(1.0, t))
            proj_lat = lat0 + t * dy
            proj_lon = lon0 + t * dx
            dev = haversine_m(lat, lon, proj_lat, proj_lon)
            max_dev = max(max_dev, dev)
        curv_max_deviation = max_dev

    return curv_accumulated, curv_max_deviation, curv_sharp_count


def parse_maxspeed(val):
    """Converte tag maxspeed para float km/h."""
    if not val:
        return None
    val = val.strip().lower()
    if val in ("none", "walk", "signals", "living_street"):
        return None
    if "mph" in val:
        try:
            return float(val.replace("mph", "").strip()) * 1.60934
        except ValueError:
            return None
    try:
        return float(val.replace("km/h", "").replace("kmh", "").strip())
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# PASSAGEM 1 - NodeDegreeCounter (SimpleHandler - rapido, sem geometria)
# ---------------------------------------------------------------------------

class NodeDegreeCounter(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.degree = defaultdict(int)
        self.n_ways = 0

    def way(self, w):
        if w.tags.get("highway", "") not in HIGHWAY_VALUES:
            return
        self.n_ways += 1
        for node_ref in w.nodes:
            self.degree[node_ref.ref] += 1


def count_node_degrees(pbf_path: str) -> dict:
    print(f"  [Passagem 1] Contando grau dos nos em {Path(pbf_path).name} ...")
    handler = NodeDegreeCounter()
    handler.apply_file(pbf_path)
    print(f"    -> {handler.n_ways:,} ways de highway processadas")
    print(f"    -> {len(handler.degree):,} nos unicos referenciados")
    return dict(handler.degree)


# ---------------------------------------------------------------------------
# PASSAGEM 2 - NodeExtractor (SimpleHandler com locacoes)
# ---------------------------------------------------------------------------

class NodeExtractor(osmium.SimpleHandler):
    def __init__(self, degree_map: dict):
        super().__init__()
        self.degree = degree_map
        self.records = []

    def node(self, n):
        if not n.location.valid():
            return
        lat = n.location.lat
        lon = n.location.lon
        osm_id = n.id

        hw = n.tags.get("highway", "")
        if hw == "traffic_signals":
            self.records.append({
                "osm_id": osm_id, "lat": lat, "lon": lon,
                "feature_type": "semaforo", "subtype": None, "extra": None,
            })
            return

        if self.degree.get(osm_id, 0) >= 3:
            self.records.append({
                "osm_id": osm_id, "lat": lat, "lon": lon,
                "feature_type": "cruzamento", "subtype": None, "extra": None,
            })

        amenity = n.tags.get("amenity", "")
        if amenity in AMENITY_NODES:
            self.records.append({
                "osm_id": osm_id, "lat": lat, "lon": lon,
                "feature_type": AMENITY_NODES[amenity], "subtype": amenity, "extra": None,
            })

        place = n.tags.get("place", "")
        if place in PLACE_VALUES:
            self.records.append({
                "osm_id": osm_id, "lat": lat, "lon": lon,
                "feature_type": "place", "subtype": place,
                "extra": n.tags.get("name", ""),
            })

    def to_dataframe(self):
        return pd.DataFrame(self.records)


def extract_nodes(pbf_path: str, degree_map: dict) -> pd.DataFrame:
    print("  [Passagem 2] Extraindo nos de interesse ...")
    handler = NodeExtractor(degree_map)
    handler.apply_file(pbf_path)
    df = handler.to_dataframe()
    print(f"    -> {len(df):,} registros de nos extraidos")
    if not df.empty:
        print(f"    -> Tipos: {df['feature_type'].value_counts().to_dict()}")
    return df


# ---------------------------------------------------------------------------
# PASSAGEM 3 - WayExtractor via FileProcessor.with_locations() (osmium 4.x)
# ---------------------------------------------------------------------------

def extract_ways(pbf_path: str) -> pd.DataFrame:
    """
    Usa FileProcessor.with_locations() para obter geometria completa das ways.
    Mais eficiente que SimpleHandler + NodeLocationsForWays na API v4.
    """
    print("  [Passagem 3] Extraindo ways com geometria ...")

    records = []
    n_no_geom = 0

    for obj in osmium.FileProcessor(pbf_path).with_locations():
        # So processar ways
        if not hasattr(obj, 'nodes'):
            continue

        hw = obj.tags.get("highway", "")
        if hw not in HIGHWAY_VALUES:
            continue

        # Reconstruir geometria
        coords = []
        for node_ref in obj.nodes:
            if node_ref.location.valid():
                coords.append((node_ref.location.lat, node_ref.location.lon))

        if len(coords) < 2:
            n_no_geom += 1
            continue

        # Comprimento
        length_m = sum(
            haversine_m(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1])
            for i in range(len(coords) - 1)
        )

        # Curvatura
        curv_acc, curv_max_dev, curv_sharp = compute_curvature(coords)

        # Tags
        maxspeed = parse_maxspeed(obj.tags.get("maxspeed", ""))
        try:
            lanes = int(obj.tags.get("lanes", "0") or 0)
        except ValueError:
            lanes = 0

        bridge       = obj.tags.get("bridge",   "no") in ("yes", "1", "true")
        tunnel       = obj.tags.get("tunnel",   "no") in ("yes", "1", "true")
        is_roundabout = obj.tags.get("junction", "") == "roundabout"
        ref  = obj.tags.get("ref",  "")
        name = obj.tags.get("name", "")

        records.append({
            "osm_id":            obj.id,
            "highway":           hw,
            "length_m":          length_m,
            "maxspeed":          maxspeed,
            "lanes":             lanes if lanes > 0 else None,
            "bridge":            bridge,
            "tunnel":            tunnel,
            "is_roundabout":     is_roundabout,
            "curv_accumulated":  curv_acc,
            "curv_max_deviation":curv_max_dev,
            "curv_sharp_count":  curv_sharp,
            "coords_json":       json.dumps(coords),
            "ref":               ref,
            "name":              name,
            "centroid_lat":      sum(c[0] for c in coords) / len(coords),
            "centroid_lon":      sum(c[1] for c in coords) / len(coords),
            "n_coords":          len(coords),
        })

    df = pd.DataFrame(records)
    print(f"    -> {len(df):,} ways extraidas")
    print(f"    -> {n_no_geom:,} ways sem geometria (ignoradas)")
    return df


# ---------------------------------------------------------------------------
# PASSAGEM 4 - AreaExtractor via FileProcessor.with_areas().with_locations()
# ---------------------------------------------------------------------------

def extract_areas(pbf_path: str) -> pd.DataFrame:
    """
    Extrai poligonos de landuse usando FileProcessor.with_areas().
    """
    print("  [Passagem 4] Extraindo areas de landuse ...")
    records = []

    for obj in osmium.FileProcessor(pbf_path).with_areas().with_locations():
        # So processar areas
        if not hasattr(obj, 'outer_rings'):
            continue

        landuse = obj.tags.get("landuse", "")
        if landuse not in LANDUSE_URBAN and landuse not in LANDUSE_RURAL:
            continue

        try:
            rings = list(obj.outer_rings())
            if not rings:
                continue

            lats = [n.location.lat for n in rings[0] if n.location.valid()]
            lons = [n.location.lon for n in rings[0] if n.location.valid()]

            if len(lats) < 3:
                continue

            centroid_lat = sum(lats) / len(lats)
            centroid_lon = sum(lons) / len(lons)

            # Estimativa de area (Shoelace em projecao planar)
            n = len(lats)
            area_deg = 0.0
            for i in range(n):
                j = (i + 1) % n
                area_deg += lats[i] * lons[j]
                area_deg -= lats[j] * lons[i]
            area_deg = abs(area_deg) / 2.0

            lat_rad = math.radians(centroid_lat)
            m_per_deg_lat = 111_132.0
            m_per_deg_lon = 111_320.0 * math.cos(lat_rad)
            area_m2 = area_deg * m_per_deg_lat * m_per_deg_lon

            landuse_class = "urban" if landuse in LANDUSE_URBAN else "rural"

            records.append({
                "osm_id":        obj.id,
                "centroid_lat":  centroid_lat,
                "centroid_lon":  centroid_lon,
                "area_m2":       area_m2,
                "landuse":       landuse,
                "landuse_class": landuse_class,
            })
        except Exception:
            continue

    df = pd.DataFrame(records)
    print(f"    -> {len(df):,} areas de landuse extraidas")
    if not df.empty:
        print(f"    -> Classes: {df['landuse_class'].value_counts().to_dict()}")
    return df


# ---------------------------------------------------------------------------
# Funcao principal
# ---------------------------------------------------------------------------

def process_pbf(pbf_path: str, region: str):
    """Processa um arquivo .pbf e salva os 3 parquets intermediarios."""

    out_nodes   = PROC_DIR / f"nodes_{region}.parquet"
    out_ways    = PROC_DIR / f"ways_{region}.parquet"
    out_landuse = PROC_DIR / f"landuse_{region}.parquet"

    if out_nodes.exists() and out_ways.exists() and out_landuse.exists():
        print(f"  [SKIP] {region} ja processado. Delete os arquivos para reprocessar.")
        return

    print(f"\n{'='*60}")
    print(f"  Processando: {region}")
    print(f"  Arquivo: {pbf_path}")
    size_mb = os.path.getsize(pbf_path) / 1024 / 1024
    print(f"  Tamanho: {size_mb:.1f} MB")
    print(f"{'='*60}")

    import time
    t0 = time.time()

    # Passagem 1
    degree_map = count_node_degrees(pbf_path)
    print(f"  Passagem 1 concluida em {time.time()-t0:.1f}s")

    # Passagem 2
    t1 = time.time()
    df_nodes = extract_nodes(pbf_path, degree_map)
    del degree_map
    print(f"  Passagem 2 concluida em {time.time()-t1:.1f}s")

    # Passagem 3
    t2 = time.time()
    df_ways = extract_ways(pbf_path)
    print(f"  Passagem 3 concluida em {time.time()-t2:.1f}s")

    # Passagem 4
    t3 = time.time()
    df_landuse = extract_areas(pbf_path)
    print(f"  Passagem 4 concluida em {time.time()-t3:.1f}s")

    # Salvar
    print(f"\n  Salvando parquets ...")
    df_nodes.to_parquet(out_nodes, index=False) if not df_nodes.empty else pd.DataFrame().to_parquet(out_nodes, index=False)
    df_ways.to_parquet(out_ways, index=False) if not df_ways.empty else pd.DataFrame().to_parquet(out_ways, index=False)
    df_landuse.to_parquet(out_landuse, index=False) if not df_landuse.empty else pd.DataFrame().to_parquet(out_landuse, index=False)

    total = time.time() - t0
    print(f"  OK {region} concluido em {total/60:.1f} min\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pbf_path", help="Caminho para o arquivo .osm.pbf")
    parser.add_argument("--region", help="Nome da regiao")
    args = parser.parse_args()

    pbf = Path(args.pbf_path)
    if not pbf.exists():
        print(f"ERRO: arquivo nao encontrado: {pbf}")
        sys.exit(1)

    region = args.region or pbf.stem.replace("-260612", "").replace("-", "_")
    process_pbf(str(pbf), region)
