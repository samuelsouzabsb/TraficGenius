import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def main():
    dataset_path = Path(r"C:\Users\samue\Documents\trafic\dataset\clima\dataset_enriquecido.parquet")
    output_path = Path(r"C:\Users\samue\Documents\trafic\dataset\clima\dataset_enriquecido_clima.parquet")
    
    us_weather_dir = Path(r"C:\Users\samue\Documents\trafic\dataset\clima\processado_us_hourly")
    br_weather_dir = Path(r"C:\Users\samue\Documents\trafic\dataset\clima\processado_br_hourly")
    
    print("1. Identificando meses de clima disponíveis...")
    us_files = sorted(list(us_weather_dir.glob("clima_h3_us_hourly_*.parquet")))
    br_files = sorted(list(br_weather_dir.glob("clima_h3_br_hourly_*.parquet")))
    
    us_available_months = {f.stem.replace("clima_h3_us_hourly_", "") for f in us_files}
    br_available_months = {f.stem.replace("clima_h3_br_hourly_", "") for f in br_files}
    
    print(f"   EUA: {len(us_available_months)} meses disponíveis ({min(us_available_months)} a {max(us_available_months)})")
    print(f"   Brasil: {len(br_available_months)} meses disponíveis ({min(br_available_months)} a {max(br_available_months)})")
    
    print("\n2. Carregando dataset enriquecido original...")
    df_accidents = pd.read_parquet(dataset_path)
    print(f"   Dataset carregado com {len(df_accidents)} linhas.")
    
    # Remove registros com valores nulos nas chaves de cruzamento temporais/espaciais
    print("   Removendo linhas com valores nulos em h3_9, data_inversa ou horario...")
    df_accidents = df_accidents.dropna(subset=["h3_9", "data_inversa", "horario"])
    print(f"   Registros válidos: {len(df_accidents)} linhas.")
    
    # Adiciona colunas auxiliares de ano e mês para filtragem
    df_accidents["year_month"] = df_accidents["data_inversa"].dt.strftime("%Y_%m")
    
    # Filtra apenas os registros que possuem correspondência na base de clima
    print("\n3. Filtrando registros com base nos meses de clima disponíveis...")
    is_us = df_accidents["pais"] == "US"
    is_br = df_accidents["pais"] == "BR"
    
    df_us_filtered = df_accidents[is_us & df_accidents["year_month"].isin(us_available_months)].copy()
    df_br_filtered = df_accidents[is_br & df_accidents["year_month"].isin(br_available_months)].copy()
    
    print(f"   EUA após filtro de meses: {len(df_us_filtered)} linhas (removidas {sum(is_us) - len(df_us_filtered)} linhas).")
    print(f"   Brasil após filtro de meses: {len(df_br_filtered)} linhas (removidas {sum(is_br) - len(df_br_filtered)} linhas, incluindo 2007-2014).")
    
    # Junta as duas bases filtradas
    df_filtered = pd.concat([df_us_filtered, df_br_filtered], ignore_index=True)
    print(f"   Total de registros a serem processados: {len(df_filtered)} linhas.")
    
    del df_accidents
    del df_us_filtered
    del df_br_filtered
    
    # Formata a coluna de tempo arredondada para hora cheia no padrão do clima
    print("\n4. Preparando coluna de tempo arredondada para o merge...")
    # Cria o datetime a partir de data_inversa e horario
    datetime_raw = pd.to_datetime(
        df_filtered["data_inversa"].dt.strftime("%Y-%m-%d") + " " + df_filtered["horario"].astype(str)
    )
    df_filtered["datetime"] = datetime_raw.dt.round("h").dt.strftime("%Y-%m-%d %H:%M:%S")
    
    # Agrupa por pais e year_month para fazer merges menores e eficientes em memória
    groups = list(df_filtered.groupby(["pais", "year_month"]))
    
    merged_chunks = []
    
    print("\n5. Mesclando dados climáticos por país e mês...")
    for (pais, ym), df_group in tqdm(groups, desc="Processando grupos"):
        # Determina o arquivo de clima correspondente
        if pais == "US":
            weather_file = us_weather_dir / f"clima_h3_us_hourly_{ym}.parquet"
        else:
            weather_file = br_weather_dir / f"clima_h3_br_hourly_{ym}.parquet"
            
        if not weather_file.exists():
            # Apenas um fallback caso não exista o arquivo físico por algum motivo
            continue
            
        # Carrega os dados climáticos
        df_weather = pd.read_parquet(weather_file)
        
        # Garante tipos de dados compatíveis para o merge
        df_weather["h3_9"] = df_weather["h3_9"].astype(str)
        df_weather["datetime"] = df_weather["datetime"].astype(str)
        
        # Merge inner para trazer as colunas climáticas
        df_merged_group = pd.merge(
            df_group,
            df_weather,
            on=["h3_9", "datetime"],
            how="inner"
        )
        
        merged_chunks.append(df_merged_group)
        
    print("\n6. Consolidando resultados e salvando...")
    if not merged_chunks:
        print("Erro: Nenhum registro mesclado com sucesso.")
        return
        
    df_final = pd.concat(merged_chunks, ignore_index=True)
    
    # Remove as colunas auxiliares temporárias
    df_final = df_final.drop(columns=["year_month", "datetime"])
    
    print(f"   Tamanho final do dataset consolidado: {len(df_final)} linhas.")
    
    # Salva o arquivo final
    df_final.to_parquet(output_path, compression="zstd", index=False)
    print(f"   Dataset consolidado salvo com sucesso em: {output_path}")

if __name__ == "__main__":
    main()
