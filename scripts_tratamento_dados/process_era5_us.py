import os
import zipfile
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configurações de caminhos
ERA5_DIR = Path(r"C:\Users\samue\Documents\trafic\dataset\clima\us")
ACCIDENTS_FILE = Path(r"C:\Users\samue\Documents\trafic\dataset\clima\dados_unificados.parquet")
OUTPUT_DIR = Path(r"C:\Users\samue\Documents\trafic\dataset\clima\processado_us_hourly")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_nc_from_zip(zip_path, filename):
    """Lê um arquivo NetCDF em memória a partir do arquivo ZIP do ERA5."""
    with zipfile.ZipFile(zip_path) as z:
        return z.read(filename)

def get_accident_keys():
    """Retorna um DataFrame com as chaves únicas (h3_9, datetime) de todos os acidentes nos EUA."""
    print("Carregando chaves (h3_9, data, hora) dos acidentes dos EUA...")
    df = pd.read_parquet(ACCIDENTS_FILE, columns=["pais", "h3_9", "data_inversa", "horario"])
    df_us = df[df["pais"] == "US"].dropna(subset=["h3_9", "data_inversa", "horario"])
    
    print("  Formatando timestamps e fazendo o arredondamento para hora cheia...")
    # Formata datas e horas
    df_us["datetime_raw"] = pd.to_datetime(df_us["data_inversa"].dt.strftime("%Y-%m-%d") + " " + df_us["horario"].astype(str))
    
    # Arredonda para a hora inteira
    df_us["datetime_rounded"] = df_us["datetime_raw"].dt.round("h")
    df_us["datetime_str"] = df_us["datetime_rounded"].dt.strftime("%Y-%m-%d %H:00:00")
    df_us["year_month"] = df_us["datetime_rounded"].dt.strftime("%Y_%m")
    
    # Mantém apenas as chaves únicas necessárias de (h3_9, datetime)
    unique_keys = df_us[["h3_9", "datetime_str", "year_month"]].drop_duplicates()
    print(f"Total de combinações únicas (H3_9, Horário): {len(unique_keys)}")
    return unique_keys

def build_spatial_mapping(unique_h3_cells, sample_ds):
    """Cria o mapeamento espacial H3_9 -> Coordenadas e Índices da Grade ERA5."""
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
    
    # Limites conhecidos do grid regular
    lat_start, lat_step = 50.0, -0.25
    lon_start, lon_step = -125.0, 0.25
    
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
    
    mapping_df.to_parquet(OUTPUT_DIR / "h3_to_era5_us_mapping.parquet")
    print(f"Mapeamento salvo com {len(mapping_df)} células.")
    return mapping_df

def process_single_month(zip_path_str, keys_month_df, mapping_df):
    """Processa um arquivo mensal do ERA5 individualmente. Função executada em processo separado."""
    zip_path = Path(zip_path_str)
    year_month = zip_path.stem.replace("era5_", "")
    output_file = OUTPUT_DIR / f"clima_h3_us_hourly_{year_month}.parquet"
    
    if output_file.exists():
        return f"Arquivo já existe: {output_file.name}"
        
    # Filtra chaves do mês atual
    keys_this_month = keys_month_df[keys_month_df["year_month"] == year_month]
    if len(keys_this_month) == 0:
        return f"Nenhum acidente registrado em {year_month}. Pulando..."
        
    needed_cells = keys_this_month["h3_9"].unique()
    mapping_filtered = mapping_df[mapping_df["h3_9"].isin(needed_cells)].copy()
    if len(mapping_filtered) == 0:
        return f"Mapeamento vazio para células do mês {year_month}. Pulando..."
        
    # Carrega NetCDFs
    data_inst = extract_nc_from_zip(zip_path, "data_stream-oper_stepType-instant.nc")
    data_acc = extract_nc_from_zip(zip_path, "data_stream-oper_stepType-accum.nc")
    
    with xr.open_dataset(data_inst) as ds_inst, xr.open_dataset(data_acc) as ds_acc:
        lat_indices = xr.DataArray(mapping_filtered["lat_idx"].values, dims="h3")
        lon_indices = xr.DataArray(mapping_filtered["lon_idx"].values, dims="h3")
        
        # Extração vetorizada rápida
        ds_inst_mapped = ds_inst[["t2m", "d2m", "sp", "u10", "v10", "tcc"]].isel(
            latitude=lat_indices,
            longitude=lon_indices
        ).load()
        
        ds_acc_mapped = ds_acc[["tp"]].isel(
            latitude=lat_indices,
            longitude=lon_indices
        ).load()
        
        # Conversão de unidades
        ds_inst_mapped["t2m"] = ds_inst_mapped["t2m"] - 273.15
        ds_inst_mapped["d2m"] = ds_inst_mapped["d2m"] - 273.15
        ds_acc_mapped["tp"] = ds_acc_mapped["tp"] * 1000
        
        times_str = pd.to_datetime(ds_inst_mapped.valid_time.values).strftime("%Y-%m-%d %H:%M:%S")
        h3_ids = mapping_filtered["h3_9"].values
        
        time_to_idx = {t: i for i, t in enumerate(times_str)}
        h3_to_idx = {h: i for i, h in enumerate(h3_ids)}
        
        clima_vals = {
            "temp": ds_inst_mapped["t2m"].values,
            "dew_point": ds_inst_mapped["d2m"].values,
            "pressure": ds_inst_mapped["sp"].values / 100.0,
            "wind_u": ds_inst_mapped["u10"].values,
            "wind_v": ds_inst_mapped["v10"].values,
            "cloud_cover": ds_inst_mapped["tcc"].values,
            "precip": ds_acc_mapped["tp"].values
        }
        
        records = []
        for row in keys_this_month.itertuples():
            h3_cell = row.h3_9
            t_str = row.datetime_str
            
            if h3_cell in h3_to_idx and t_str in time_to_idx:
                h_i = h3_to_idx[h3_cell]
                t_i = time_to_idx[t_str]
                
                records.append({
                    "datetime": t_str,
                    "h3_9": h3_cell,
                    "temperature_c": clima_vals["temp"][t_i, h_i],
                    "dew_point_c": clima_vals["dew_point"][t_i, h_i],
                    "pressure_hpa": clima_vals["pressure"][t_i, h_i],
                    "wind_u": clima_vals["wind_u"][t_i, h_i],
                    "wind_v": clima_vals["wind_v"][t_i, h_i],
                    "cloud_cover": clima_vals["cloud_cover"][t_i, h_i],
                    "precip_mm": clima_vals["precip"][t_i, h_i]
                })
                
        if len(records) > 0:
            df_out = pd.DataFrame(records)
            df_out.to_parquet(output_file, compression="zstd")
            return f"Processado com sucesso: {output_file.name} ({len(df_out)} linhas)"
        else:
            return f"Sem registros coincidentes em {year_month}."

def main():
    # 1. Carrega chaves de acidentes
    keys_df = get_accident_keys()
    unique_cells = keys_df["h3_9"].unique()
    
    # 2. Carrega uma grade amostra do ERA5 para mapeamento espacial
    all_zips = sorted(list(ERA5_DIR.glob("*.nc")))
    if not all_zips:
        print("Nenhum arquivo NetCDF encontrado na pasta dos EUA.")
        return
        
    first_zip = all_zips[0]
    data_sample = extract_nc_from_zip(first_zip, "data_stream-oper_stepType-instant.nc")
    
    with xr.open_dataset(data_sample) as sample_ds:
        mapping_path = OUTPUT_DIR / "h3_to_era5_us_mapping.parquet"
        if mapping_path.exists():
            print("Carregando mapeamento espacial existente...")
            mapping_df = pd.read_parquet(mapping_path)
        else:
            mapping_df = build_spatial_mapping(unique_cells, sample_ds)
            
    # 3. Paralelização usando ProcessPoolExecutor
    # 24 núcleos disponíveis. Vamos usar cerca de 6 a 8 processos concorrentes
    # para não ter contenção de memória RAM, já que abrir múltiplos datasets
    # de xarray consome bastante memória.
    max_workers = 6
    print(f"Iniciando pool de processos paralelos com {max_workers} trabalhadores...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for zip_path in all_zips:
            futures[executor.submit(process_single_month, str(zip_path), keys_df, mapping_df)] = zip_path.name
            
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processando meses"):
            zip_name = futures[future]
            try:
                result = future.result()
                print(f"\n[{zip_name}] {result}")
            except Exception as e:
                print(f"\nErro ao processar {zip_name}: {e}")

if __name__ == "__main__":
    main()
