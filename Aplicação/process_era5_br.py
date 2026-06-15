import os
import zipfile
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configurações de caminhos
ERA5_DIR = Path(r"C:\Users\samue\Documents\trafic\dataset\clima\brasil")
ACCIDENTS_FILE = Path(r"C:\Users\samue\Documents\trafic\dataset\clima\dados_unificados.parquet")
OUTPUT_DIR = Path(r"C:\Users\samue\Documents\trafic\dataset\clima\processado_br_hourly")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_nc_from_zip(zip_path, filename):
    """Lê um arquivo NetCDF em memória a partir do arquivo ZIP do ERA5."""
    with zipfile.ZipFile(zip_path) as z:
        return z.read(filename)

def get_accident_keys():
    """Retorna um DataFrame com as chaves únicas (h3_9, datetime) de todos os acidentes no Brasil (2015 em diante)."""
    print("Carregando chaves (h3_9, data, hora) dos acidentes do Brasil...")
    df = pd.read_parquet(ACCIDENTS_FILE, columns=["pais", "h3_9", "data_inversa", "horario"])
    # Filtra apenas o Brasil e garante que o ano seja >= 2015
    df_br = df[(df["pais"] == "BR") & (df["data_inversa"].dt.year >= 2015)].dropna(subset=["h3_9", "data_inversa", "horario"])
    
    print("  Formatando timestamps e fazendo o arredondamento para hora cheia...")
    df_br["datetime_raw"] = pd.to_datetime(df_br["data_inversa"].dt.strftime("%Y-%m-%d") + " " + df_br["horario"].astype(str))
    df_br["datetime_rounded"] = df_br["datetime_raw"].dt.round("h")
    # Mantém como objeto datetime para facilitar a vetorização do Pandas no Merge final
    df_br["datetime"] = df_br["datetime_rounded"]
    df_br["year_month"] = df_br["datetime"].dt.strftime("%Y_%m")
    
    unique_keys = df_br[["h3_9", "datetime", "year_month"]].drop_duplicates()
    print(f"Total de combinações únicas no Brasil (H3_9, Horário): {len(unique_keys)}")
    return unique_keys

def build_spatial_mapping(unique_h3_cells, sample_ds):
    """Cria o mapeamento espacial H3_9 -> Coordenadas e Índices da Grade ERA5 para o Brasil."""
    print("Construindo mapeamento espacial (H3_9 -> Índices Grade ERA5)...")
    import h3
    
    era5_lats = sample_ds.latitude.values
    era5_lons = sample_ds.longitude.values
    
    h3_coords = []
    for cell in tqdm(unique_h3_cells, desc="Obtendo coordenadas de células"):
        try:
            lat, lon = h3.cell_to_latlng(cell)
            h3_coords.append((cell, lat, lon))
        except Exception:
            pass
            
    mapping_df = pd.DataFrame(h3_coords, columns=["h3_9", "h3_lat", "h3_lon"])
    
    # Limites conhecidos do grid regular do Brasil
    lat_start, lat_step = 6.0, -0.25
    lon_start, lon_step = -75.0, 0.25
    
    lat_indices = []
    lon_indices = []
    
    for _, row in tqdm(mapping_df.iterrows(), total=len(mapping_df), desc="Mapeando índices de grade"):
        lat_idx = int(round((row["h3_lat"] - lat_start) / lat_step))
        lon_idx = int(round((row["h3_lon"] - lon_start) / lon_step))
        
        # Garante limites
        lat_idx = max(0, min(len(era5_lats) - 1, lat_idx))
        lon_idx = max(0, min(len(era5_lons) - 1, lon_idx))
        
        lat_indices.append(lat_idx)
        lon_indices.append(lon_idx)
        
    mapping_df["lat_idx"] = lat_indices
    mapping_df["lon_idx"] = lon_indices
    
    mapping_df.to_parquet(OUTPUT_DIR / "h3_to_era5_br_mapping.parquet")
    print(f"Mapeamento salvo com {len(mapping_df)} células.")
    return mapping_df

def process_single_month(zip_path_str, keys_this_month):
    """Processa um arquivo mensal usando merge vetorizado puro do Pandas (muito mais rápido!)."""
    zip_path = Path(zip_path_str)
    year_month = zip_path.stem.replace("era5BR_", "")
    output_file = OUTPUT_DIR / f"clima_h3_br_hourly_{year_month}.parquet"
    
    if output_file.exists():
        return f"Arquivo já existe: {output_file.name}"
        
    if keys_this_month is None or len(keys_this_month) == 0:
        return f"Nenhum acidente registrado em {year_month}. Pulando..."
        
    needed_cells = keys_this_month["h3_9"].unique()
    
    # Carrega o mapeamento espacial direto do disco para evitar overhead de IPC no Windows
    mapping_path = OUTPUT_DIR / "h3_to_era5_br_mapping.parquet"
    mapping_df = pd.read_parquet(mapping_path)
    
    mapping_filtered = mapping_df[mapping_df["h3_9"].isin(needed_cells)].copy()
    if len(mapping_filtered) == 0:
        return f"Mapeamento vazio para células do mês {year_month}. Pulando..."
        
    # Carrega NetCDFs
    data_inst = extract_nc_from_zip(zip_path, "data_stream-oper_stepType-instant.nc")
    data_acc = extract_nc_from_zip(zip_path, "data_stream-oper_stepType-accum.nc")
    
    with xr.open_dataset(data_inst).load() as ds_inst, xr.open_dataset(data_acc).load() as ds_acc:
        lat_indices = xr.DataArray(mapping_filtered["lat_idx"].values, dims="h3")
        lon_indices = xr.DataArray(mapping_filtered["lon_idx"].values, dims="h3")
        
        # Extração vetorizada rápida (já em memória)
        ds_inst_mapped = ds_inst[["t2m", "d2m", "sp", "u10", "v10", "tcc"]].isel(
            latitude=lat_indices,
            longitude=lon_indices
        )
        
        ds_acc_mapped = ds_acc[["tp"]].isel(
            latitude=lat_indices,
            longitude=lon_indices
        )
        
        # Conversão de unidades
        ds_inst_mapped["t2m"] = ds_inst_mapped["t2m"] - 273.15
        ds_inst_mapped["d2m"] = ds_inst_mapped["d2m"] - 273.15
        ds_acc_mapped["tp"] = ds_acc_mapped["tp"] * 1000
        
        # Criação de um DataFrame completo com os dados climáticos mapeados para todos os H3
        df_inst = ds_inst_mapped.to_dataframe().reset_index()
        df_acc = ds_acc_mapped.to_dataframe().reset_index()
        
        # Faz o merge local dos instantâneos com os acumulados por valid_time e índice h3
        df_clima_mapped = pd.merge(df_inst, df_acc, on=["valid_time", "h3"])
        
        # Substitui o índice de dimensão 'h3' temporário pela tag H3_9 correspondente
        h3_ids = mapping_filtered["h3_9"].values
        df_clima_mapped["h3_9"] = df_clima_mapped["h3"].apply(lambda idx: h3_ids[idx])
        
        # Renomeia colunas para o padrão dos EUA
        df_clima_mapped = df_clima_mapped.rename(columns={
            "valid_time": "datetime",
            "t2m": "temperature_c",
            "d2m": "dew_point_c",
            "sp": "pressure_hpa",
            "u10": "wind_u",
            "v10": "wind_v",
            "tcc": "cloud_cover",
            "tp": "precip_mm"
        })
        
        # Ajusta unidades de pressão
        df_clima_mapped["pressure_hpa"] = df_clima_mapped["pressure_hpa"] / 100.0
        
        # Mantém apenas as variáveis necessárias
        df_clima_mapped = df_clima_mapped[[
            "datetime", "h3_9", "temperature_c", "dew_point_c", 
            "pressure_hpa", "wind_u", "wind_v", "cloud_cover", "precip_mm"
        ]]
        
        # MERGE VETORIZADO SUPER RÁPIDO DO PANDAS!
        # Cruzamos apenas as chaves reais de acidentes daquele mês com o grid clima gerado
        df_out = pd.merge(
            keys_this_month[["h3_9", "datetime"]],
            df_clima_mapped,
            on=["h3_9", "datetime"],
            how="inner"
        )
        
        # Formata a coluna temporal como string para salvar
        df_out["datetime"] = df_out["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        
        if len(df_out) > 0:
            df_out.to_parquet(output_file, compression="zstd")
            return f"Processado com sucesso: {output_file.name} ({len(df_out)} linhas)"
        else:
            return f"Sem registros coincidentes em {year_month}."

def main():
    # 1. Carrega chaves de acidentes BR
    keys_df = get_accident_keys()
    unique_cells = keys_df["h3_9"].unique()
    
    # 2. Carrega uma grade amostra do ERA5 para mapeamento espacial
    all_zips = sorted(list(ERA5_DIR.glob("*.nc")))
    if not all_zips:
        print("Nenhum arquivo NetCDF do Brasil encontrado.")
        return
        
    first_zip = all_zips[0]
    data_sample = extract_nc_from_zip(first_zip, "data_stream-oper_stepType-instant.nc")
    
    with xr.open_dataset(data_sample) as sample_ds:
        mapping_path = OUTPUT_DIR / "h3_to_era5_br_mapping.parquet"
        if mapping_path.exists():
            print("Carregando mapeamento espacial existente...")
            mapping_df = pd.read_parquet(mapping_path)
        else:
            mapping_df = build_spatial_mapping(unique_cells, sample_ds)
            
    # 3. Paralelização usando ProcessPoolExecutor com 8 Workers
    max_workers = 8
    print(f"Iniciando pool de processos paralelos com {max_workers} trabalhadores...")
    
    # Agrupa e projeta as chaves de acidente por mês para evitar trafegar um dataframe gigante por IPC no Windows
    print("Agrupando chaves de acidente por mês para otimização de IPC...")
    keys_by_month = {}
    for ym, group in keys_df.groupby("year_month"):
        keys_by_month[ym] = group[["h3_9", "datetime"]].copy()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for zip_path in all_zips:
            year_month = zip_path.stem.replace("era5BR_", "")
            keys_this_month = keys_by_month.get(year_month)
            futures[executor.submit(process_single_month, str(zip_path), keys_this_month)] = zip_path.name
            
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processando meses BR"):
            zip_name = futures[future]
            try:
                result = future.result()
                print(f"\n[{zip_name}] {result}")
            except Exception as e:
                print(f"\nErro ao processar {zip_name}: {e}")

if __name__ == "__main__":
    main()
