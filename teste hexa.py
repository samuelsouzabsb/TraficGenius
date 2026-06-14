import osmnx as ox
import pandas as pd
import numpy as np

# ==========================================
# COORDENADA (ou centro do H3)
# ==========================================

lat = -15.6881578899999
lon = -47.85885178

lat = -20.8309863894233
lon = -44.8223217699038

RAIO_VIAS = 500
RAIO_POIS = 3000

features = {}

# ==========================================
# REDE VIÁRIA
# ==========================================

try:

    G = ox.graph_from_point(
        (lat, lon),
        dist=RAIO_VIAS,
        network_type="drive"
    )

    edges = ox.graph_to_gdfs(
        G,
        nodes=False
    )

    # ------------------------------
    # Estrutura da rede
    # ------------------------------

    features["num_nos"] = len(G.nodes)
    features["num_arestas"] = len(G.edges)

    features["num_intersecoes"] = len(
        [n for n, d in G.degree() if d > 2]
    )

    features["comprimento_total_vias_m"] = (
        edges["length"].sum()
    )

    # ------------------------------
    # Tipos de rodovia
    # ------------------------------

    if "highway" in edges.columns:

        highway_series = (
            edges["highway"]
            .explode()
            .astype(str)
        )

        tipos = [
            "motorway",
            "trunk",
            "primary",
            "secondary",
            "tertiary",
            "residential",
            "service",
            "unclassified"
        ]

        for tipo in tipos:

            features[f"road_{tipo}"] = (
                highway_series == tipo
            ).sum()

    # ------------------------------
    # Velocidade
    # ------------------------------

    velocidades = []

    if "maxspeed" in edges.columns:

        for v in edges["maxspeed"].dropna():

            try:

                if isinstance(v, list):
                    v = v[0]

                velocidades.append(
                    float(str(v).split()[0])
                )

            except:
                pass

    if len(velocidades):

        features["velocidade_media"] = np.mean(
            velocidades
        )

        features["velocidade_max"] = np.max(
            velocidades
        )

    # ------------------------------
    # Faixas
    # ------------------------------

    faixas = []

    if "lanes" in edges.columns:

        for l in edges["lanes"].dropna():

            try:

                if isinstance(l, list):
                    l = l[0]

                faixas.append(float(l))

            except:
                pass

    if len(faixas):

        features["faixas_media"] = np.mean(
            faixas
        )

        features["faixas_max"] = np.max(
            faixas
        )

    # ------------------------------
    # Mão única
    # ------------------------------

    if "oneway" in edges.columns:

        features["vias_mao_unica"] = (
            edges["oneway"] == True
        ).sum()

    # ------------------------------
    # Ponte
    # ------------------------------

    if "bridge" in edges.columns:

        features["pontes"] = (
            edges["bridge"].notna()
        ).sum()

    # ------------------------------
    # Túnel
    # ------------------------------

    if "tunnel" in edges.columns:

        features["tuneis"] = (
            edges["tunnel"].notna()
        ).sum()

    # ------------------------------
    # Rotatória
    # ------------------------------

    if "junction" in edges.columns:

        features["rotatorias"] = (
            edges["junction"] == "roundabout"
        ).sum()

    # ------------------------------
    # Iluminação
    # ------------------------------

    if "lit" in edges.columns:

        features["vias_iluminadas"] = (
            edges["lit"] == "yes"
        ).sum()

except Exception as e:

    print("Erro na rede viária:", e)

# ==========================================
# POIs (Pontos de Interesse)
# ==========================================

try:

    tags = {
        "amenity": True,
        "shop": True,
        "tourism": True,
        "railway": True
    }

    pois = ox.features_from_point(
        (lat, lon),
        tags=tags,
        dist=RAIO_POIS
    )

    # ------------------------------
    # Amenity
    # ------------------------------

    if "amenity" in pois.columns:

        amenities = pois["amenity"].astype(str)

        features["hospitais"] = (
            amenities == "hospital"
        ).sum()

        features["postos_saude"] = (
            amenities == "clinic"
        ).sum()

        features["escolas"] = (
            amenities == "school"
        ).sum()

        features["universidades"] = (
            amenities == "university"
        ).sum()

        features["postos_combustivel"] = (
            amenities == "fuel"
        ).sum()

        features["restaurantes"] = (
            amenities == "restaurant"
        ).sum()

        features["bancos"] = (
            amenities == "bank"
        ).sum()

        features["estacionamentos"] = (
            amenities == "parking"
        ).sum()

    # ------------------------------
    # Comércio
    # ------------------------------

    if "shop" in pois.columns:

        features["quantidade_lojas"] = (
            pois["shop"].notna()
        ).sum()

    # ------------------------------
    # Turismo
    # ------------------------------

    if "tourism" in pois.columns:

        features["locais_turisticos"] = (
            pois["tourism"].notna()
        ).sum()

    # ------------------------------
    # Ferrovia
    # ------------------------------

    if "railway" in pois.columns:

        features["ferrovias"] = (
            pois["railway"].notna()
        ).sum()

except Exception as e:

    print("Erro nos POIs:", e)

# ==========================================
# RESULTADO
# ==========================================

resultado = (
    pd.DataFrame([features])
    .T
    .reset_index()
)

resultado.columns = [
    "variavel",
    "valor"
]

print(resultado)


"""import h3

# Coordenadas
lat = float(str("-20,830986").replace(",", "."))
lon = float(str("-44,822321").replace(",", "."))


import osmnx as ox

lat = -20.8309863894233
lon = -44.8223217699038

try:
    G = ox.graph_from_point(
        (lat, lon),
        dist=1500,
        network_type="drive"
    )
    print(
    len(
        [n for n,d in G.degree() if d > 2]
    )
    )
    tags = {
    "highway": True
    }

    vias = ox.features_from_point(
    (lat, lon),
    tags=tags,
    dist=100
    )

    print(vias["highway"].value_counts())

    print(f"Nós: {len(G.nodes)}")
    print(f"Arestas: {len(G.edges)}")

except Exception as e:
    print(e)
""""""
G = ox.graph_from_point(
    (lat, lon),
    dist=500,
    network_type="all"
)

print(
    len(
        [n for n,d in G.degree() if d > 2]
    )
)
tags = {
    "highway": True
}

vias = ox.features_from_point(
    (lat, lon),
    tags=tags,
    dist=100
)

print(vias["highway"].value_counts())








# Resolução H3 (8 é uma boa escolha para acidentes)
resolucao = 8

# Hexágono
hex_id = h3.latlng_to_cell(lat, lon, resolucao)

print(f"Hexágono H3: {hex_id}")

# Centro do hexágono
lat_centro, lon_centro = h3.cell_to_latlng(hex_id)

print(f"Centro: {lat_centro}, {lon_centro}")

# Área do hexágono
area = h3.cell_area(hex_id, unit="km^2")

print(f"Área: {area:.4f} km²")

# Vizinhos imediatos
vizinhos = h3.grid_disk(hex_id, 1)

print(f"Quantidade de células (centro + vizinhos): {len(vizinhos)}")

print("\nVizinhos:")
for h in vizinhos:
    print(h)

# Vértices do hexágono
vertices = h3.cell_to_boundary(hex_id)

print("\nVértices:")
for lat_v, lon_v in vertices:
    print(f"{lat_v}, {lon_v}")"""