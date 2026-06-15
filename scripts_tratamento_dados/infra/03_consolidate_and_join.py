"""
03_consolidate_and_join.py
==========================
Consolida todos os parquets H3 de features (de todos os estados/regiões)
em 3 arquivos globais e faz o join com a base de acidentes.

Saída:
    dataset/dataset_enriquecido.parquet
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parents[2]
FEAT_DIR   = BASE_DIR / "dataset" / "infra" / "h3_features"
ACCIDENTS  = BASE_DIR / "dataset" / "dados_unificados.parquet"
OUTPUT     = BASE_DIR / "dataset" / "dataset_enriquecido.parquet"


# ---------------------------------------------------------------------------
# Funções de consolidação
# ---------------------------------------------------------------------------

def safe_mode(series):
    m = series.dropna().mode()
    return m.iloc[0] if not m.empty else None


def consolidate_h3_11(feat_dir: Path, valid_cells: set) -> pd.DataFrame:
    """Consolida todos os parquets h3_11_*.parquet em um único DataFrame filtrando H3s válidos."""
    files = list((feat_dir / "h3_11").glob("h3_11_*.parquet"))
    if not files:
        print("  AVISO: nenhum arquivo h3_11_*.parquet encontrado!")
        return pd.DataFrame()

    print(f"  Consolidando e filtrando {len(files)} arquivos H3-11 ...")
    dfs = []
    for f in tqdm(files, leave=False):
        try:
            df_region = pd.read_parquet(f)
            if not df_region.empty:
                df_region = df_region[df_region["h3_11"].isin(valid_cells)]
                if not df_region.empty:
                    dfs.append(df_region)
        except Exception as e:
            print(f"  Erro ao ler {f.name}: {e}")
            
    if not dfs:
        print("  AVISO: nenhuma célula de H3-11 corresponde aos acidentes.")
        return pd.DataFrame()
        
    df = pd.concat(dfs, ignore_index=True)
    print(f"    -> {len(df):,} linhas antes da deduplicação")

    # Deduplicar: células que aparecem em múltiplos arquivos (fronteiras)
    df = df.groupby("h3_11", as_index=False).agg({
        "n_cruzamentos":     "sum",
        "n_semaforos":       "sum",
        "speed_mean":        "mean",
        "speed_min":         "min",
        "speed_max":         "max",
        "lanes_mean":        "mean",
        "curv_accumulated":  "mean",
        "curv_max_deviation":"max",
        "curv_sharp_count":  "sum",
        "n_rotatorias":      "sum",
        "n_pontes":          "sum",
        "bridge_length_m":   "sum",
        "n_tuneis":          "sum",
        "tunnel_length_m":   "sum",
        "road_length_m":     "sum",
        "dominant_highway":  safe_mode,
    })
    df = df.rename(columns={
        "road_length_m": "road_length_m_h11",
        "dominant_highway": "dominant_highway_h11",
    })
    print(f"    -> {len(df):,} células únicas H3-11 após deduplicação")
    return df


def consolidate_h3_10(feat_dir: Path, valid_cells: set) -> pd.DataFrame:
    files = list((feat_dir / "h3_10").glob("h3_10_*.parquet"))
    if not files:
        print("  AVISO: nenhum arquivo h3_10_*.parquet encontrado!")
        return pd.DataFrame()

    print(f"  Consolidando e filtrando {len(files)} arquivos H3-10 ...")
    dfs = []
    for f in tqdm(files, leave=False):
        try:
            df_region = pd.read_parquet(f)
            if not df_region.empty:
                df_region = df_region[df_region["h3_10"].isin(valid_cells)]
                if not df_region.empty:
                    dfs.append(df_region)
        except Exception as e:
            print(f"  Erro ao ler {f.name}: {e}")
            
    if not dfs:
        print("  AVISO: nenhuma célula de H3-10 corresponde aos acidentes.")
        return pd.DataFrame()
        
    df = pd.concat(dfs, ignore_index=True)
    print(f"    -> {len(df):,} linhas antes da deduplicação")

    df = df.groupby("h3_10", as_index=False).agg({
        "n_postos":        "sum",
        "n_restaurantes":  "sum",
        "n_escolas":       "sum",
        "road_length_m":   "sum",
        "dominant_highway":safe_mode,
    })
    df = df.rename(columns={
        "road_length_m": "road_length_m_h10",
        "dominant_highway": "dominant_highway_h10",
    })
    print(f"    -> {len(df):,} células únicas H3-10 após deduplicação")
    return df


def consolidate_h3_9(feat_dir: Path, valid_cells: set) -> pd.DataFrame:
    files = list((feat_dir / "h3_9").glob("h3_9_*.parquet"))
    if not files:
        print("  AVISO: nenhum arquivo h3_9_*.parquet encontrado!")
        return pd.DataFrame()

    print(f"  Consolidando e filtrando {len(files)} arquivos H3-9 ...")
    dfs = []
    for f in tqdm(files, leave=False):
        try:
            df_region = pd.read_parquet(f)
            if not df_region.empty:
                df_region = df_region[df_region["h3_9"].isin(valid_cells)]
                if not df_region.empty:
                    dfs.append(df_region)
        except Exception as e:
            print(f"  Erro ao ler {f.name}: {e}")
            
    if not dfs:
        print("  AVISO: nenhuma célula de H3-9 corresponde aos acidentes.")
        return pd.DataFrame()
        
    df = pd.concat(dfs, ignore_index=True)
    print(f"    -> {len(df):,} linhas antes da deduplicação")

    agg_dict = {
        "n_hospitais":         "sum",
        "road_count_distinct": "sum",
        "total_sharp_curves":  "sum",
        "place_count":         "sum",
        "urban_area_m2":       "sum",
        "rural_area_m2":       "sum",
    }
    # Colunas opcionais
    for col, func in [
        ("road_total_length_m", "sum"),
        ("place_type",          "max"),
        ("dominant_highway_r9", safe_mode),
    ]:
        if col in df.columns:
            agg_dict[col] = func

    df = df.groupby("h3_9", as_index=False).agg(agg_dict)

    # Recalcular urban_ratio após consolidação
    df["urban_ratio"] = df["urban_area_m2"] / (df["urban_area_m2"] + df["rural_area_m2"] + 1e-9)
    df["urban_ratio"] = df["urban_ratio"].where(
        (df["urban_area_m2"] + df["rural_area_m2"]) > 0, other=None
    )

    # Renomear colunas para evitar colisão e manter consistência
    rename_cols = {}
    if "road_total_length_m" in df.columns:
        rename_cols["road_total_length_m"] = "road_length_m_h9"
    if "dominant_highway_r9" in df.columns:
        rename_cols["dominant_highway_r9"] = "dominant_highway_h9"
    if rename_cols:
        df = df.rename(columns=rename_cols)

    print(f"    -> {len(df):,} células únicas H3-9 após deduplicação")
    return df


# ---------------------------------------------------------------------------
# Join final
# ---------------------------------------------------------------------------

def join_with_accidents(
    accidents_path: Path,
    df_11: pd.DataFrame,
    df_10: pd.DataFrame,
    df_9:  pd.DataFrame,
    output_path: Path,
):
    print(f"\n  Carregando base de acidentes ...")
    accidents = pd.read_parquet(accidents_path)
    print(f"    -> {len(accidents):,} registros | {accidents.shape[1]} colunas")

    print(f"\n  Fazendo join H3-11 ...")
    if not df_11.empty:
        result = accidents.merge(df_11, on="h3_11", how="left")
        print(f"    -> Cobertura H3-11: {result['n_cruzamentos'].notna().mean()*100:.1f}% dos acidentes")
    else:
        result = accidents.copy()
        print("    -> Cobertura H3-11: 0.0% (sem dados)")

    print(f"  Fazendo join H3-10 ...")
    if not df_10.empty:
        result = result.merge(df_10, on="h3_10", how="left")
        print(f"    -> Cobertura H3-10: {result['n_postos'].notna().mean()*100:.1f}% dos acidentes")
    else:
        print("    -> Cobertura H3-10: 0.0% (sem dados)")

    print(f"  Fazendo join H3-9 ...")
    if not df_9.empty:
        result = result.merge(df_9, on="h3_9", how="left")
        print(f"    -> Cobertura H3-9: {result['n_hospitais'].notna().mean()*100:.1f}% dos acidentes")
    else:
        print("    -> Cobertura H3-9: 0.0% (sem dados)")

    assert len(result) == len(accidents), "Erro: Join alterou número de linhas!"

    print(f"\n  Salvando resultado em {output_path} ...")
    result.to_parquet(output_path, index=False, compression="snappy")

    size_gb = output_path.stat().st_size / 1024**3
    print(f"    -> {len(result):,} linhas × {result.shape[1]} colunas")
    print(f"    -> Tamanho: {size_gb:.2f} GB")

    return result


# ---------------------------------------------------------------------------
# Relatório de cobertura
# ---------------------------------------------------------------------------

def coverage_report(result: pd.DataFrame):
    print(f"\n{'='*60}")
    print("  RELATÓRIO DE COBERTURA")
    print(f"{'='*60}")

    feature_groups = {
        "H3-11 — Cruzamentos": "n_cruzamentos",
        "H3-11 — Semáforos":   "n_semaforos",
        "H3-11 — Velocidade":  "speed_mean",
        "H3-11 — Faixas":      "lanes_mean",
        "H3-11 — Curvatura":   "curv_accumulated",
        "H3-11 — Pontes":      "n_pontes",
        "H3-11 — Túneis":      "n_tuneis",
        "H3-11 — Dens. viária":  "road_length_m_h11",
        "H3-11 — Via dominante": "dominant_highway_h11",
        "H3-10 — Postos":      "n_postos",
        "H3-10 — Restaurantes":"n_restaurantes",
        "H3-10 — Escolas":     "n_escolas",
        "H3-10 — Dens. viária":  "road_length_m_h10",
        "H3-10 — Via dominante": "dominant_highway_h10",
        "H3-9  — Hospitais":   "n_hospitais",
        "H3-9  — Urban ratio": "urban_ratio",
        "H3-9  — Place type":  "place_type",
        "H3-9  — Dens. viária":  "road_length_m_h9",
        "H3-9  — Via dominante": "dominant_highway_h9",
    }

    for label, col in feature_groups.items():
        if col in result.columns:
            pct = result[col].notna().mean() * 100
            print(f"  {label:<30}: {pct:5.1f}% preenchido")
        else:
            print(f"  {label:<30}: coluna ausente")

    if 'pais' in result.columns:
        print(f"\n  Brasil: {(result['pais']=='BR').sum():,} acidentes")
        print(f"  EUA:    {(result['pais']=='US').sum():,} acidentes")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Consolida features H3 e faz join com a base de acidentes"
    )
    parser.add_argument(
        "--accidents", default=str(ACCIDENTS),
        help="Caminho para dados_unificados.parquet"
    )
    parser.add_argument(
        "--output", default=str(OUTPUT),
        help="Caminho de saída do dataset enriquecido"
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  CONSOLIDAÇÃO GLOBAL E JOIN FINAL")
    print(f"{'='*60}\n")

    # Carregar H3s válidos dos acidentes para filtrar na leitura (evita OOM)
    print("  Carregando H3s válidos da base de acidentes para filtrar...")
    acc_h3 = pd.read_parquet(args.accidents, columns=["h3_9", "h3_10", "h3_11"])
    valid_11 = set(acc_h3["h3_11"].dropna().unique())
    valid_10 = set(acc_h3["h3_10"].dropna().unique())
    valid_9  = set(acc_h3["h3_9"].dropna().unique())
    del acc_h3
    print(f"    -> H3-11 únicos: {len(valid_11):,}")
    print(f"    -> H3-10 únicos: {len(valid_10):,}")
    print(f"    -> H3-9  únicos: {len(valid_9):,}")

    # Consolidar features por resolução
    df_11 = consolidate_h3_11(FEAT_DIR, valid_11)
    df_10 = consolidate_h3_10(FEAT_DIR, valid_10)
    df_9  = consolidate_h3_9(FEAT_DIR, valid_9)

    # Join com acidentes
    result = join_with_accidents(
        Path(args.accidents), df_11, df_10, df_9, Path(args.output)
    )

    # Relatório
    coverage_report(result)

    print(f"\n  OK Dataset enriquecido salvo em: {args.output}\n")


if __name__ == "__main__":
    main()
