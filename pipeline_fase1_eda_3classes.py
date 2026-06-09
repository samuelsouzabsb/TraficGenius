# -*- coding: utf-8 -*-
"""
Pipeline Fase 1: Análise Exploratória de Dados (EDA) e Limpeza (3 Classes)
"""

import pandas as pd
import numpy as np
import glob
import os
import warnings
from scipy.stats import chi2
from sklearn.cluster import MiniBatchKMeans

warnings.filterwarnings('ignore')

COLUMNS_MAPPING = {
    'Severity': 'Severidade',
    'Start_Time': 'Tempo_Inicial',
    'Start_Lat': 'Latitude_Inicial',
    'Start_Lng': 'Longitude_Inicial',
    'Distance(mi)': 'Distancia_Milhas',
    'Temperature(F)': 'Temperatura_F',
    'Wind_Chill(F)': 'Sensacao_Termica_F',
    'Humidity(%)': 'Umidade_Percentual',
    'Pressure(in)': 'Pressao_Polegadas',
    'Visibility(mi)': 'Visibilidade_Milhas',
    'Wind_Speed(mph)': 'Velocidade_Vento_Mph',
    'Precipitation(in)': 'Precipitacao_Polegadas',
    'Amenity': 'Comodidade',
    'Bump': 'Lombada',
    'Crossing': 'Cruzamento',
    'Give_Way': 'Preferencia',
    'Junction': 'Juncao',
    'No_Exit': 'Sem_Saida',
    'Railway': 'Via_Ferrea',
    'Roundabout': 'Rotatoria',
    'Station': 'Estacao',
    'Stop': 'Pare',
    'Traffic_Calming': 'Redutor_Velocidade',
    'Traffic_Signal': 'Semaforo',
    'Sunrise_Sunset': 'Nascer_Por_Sol'
}

def rename_columns(df):
    print("Renomeando colunas do dataset para o português...")
    return df.rename(columns=COLUMNS_MAPPING)


def load_and_sample_data(folder_path, sample_size=None):
    import pyarrow.parquet as pq
    import random
    
    print("--- 1. Carregamento e Amostragem (3 Classes) ---")
    consolidated_file = os.path.join(folder_path, "dataset_consolidado.parquet")
    
    usecols = [
        'Severity', 'Start_Time', 'Start_Lat', 'Start_Lng', 'Distance(mi)',
        'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 
        'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)',
        'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit',
        'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal',
        'Sunrise_Sunset'
    ]
    
    if sample_size is None:
        if os.path.exists(consolidated_file):
            print(f"Lendo base consolidada completa: {os.path.basename(consolidated_file)}")
            df_sample = pd.read_parquet(consolidated_file, columns=usecols)
        else:
            print("Base consolidada não encontrada. Lendo arquivos fragmentados...")
            files = sorted(glob.glob(os.path.join(folder_path, "train-*.parquet")))
            if not files:
                raise FileNotFoundError(f"Nenhum arquivo Parquet de treino encontrado em: {folder_path}")
            dfs = [pd.read_parquet(f, columns=usecols) for f in files]
            df_sample = pd.concat(dfs, ignore_index=True)
    else:
        row_groups_info = []
        if os.path.exists(consolidated_file):
            pf = pq.ParquetFile(consolidated_file)
            for i in range(pf.num_row_groups):
                row_groups_info.append((consolidated_file, i))
        else:
            files = glob.glob(os.path.join(folder_path, "train-*.parquet"))
            for f in files:
                pf = pq.ParquetFile(f)
                for i in range(pf.num_row_groups):
                    row_groups_info.append((f, i))
                
        random.seed(42)
        random.shuffle(row_groups_info)
        dfs = []
        current_size = 0
        for f, i in row_groups_info:
            pf = pq.ParquetFile(f)
            df_part = pf.read_row_group(i, columns=usecols).to_pandas()
            dfs.append(df_part)
            current_size += len(df_part)
            if current_size >= sample_size:
                break
        df_sample = pd.concat(dfs, ignore_index=True)
        if len(df_sample) > sample_size:
            df_sample = df_sample.sample(n=sample_size, random_state=42)
            
    df_sample = rename_columns(df_sample)
    return df_sample

def map_target_to_3classes(df):
    """
    Mapeia a severidade original de 4 classes para a nova escala de 3 classes:
    - 1 e 2 -> 1 (Leve/Médio)
    - 3     -> 2 (Grave)
    - 4     -> 3 (Fatal)
    """
    print("Mapeando target de 4 classes para 3 classes...")
    df['Severidade'] = df['Severidade'].map({1: 1, 2: 1, 3: 2, 4: 3})
    return df

def feature_engineering(df):
    print("\n--- 2. Feature Engineering (3 Classes) ---")
    df['Tempo_Inicial'] = pd.to_datetime(df['Tempo_Inicial'], format='mixed')
    df['Hora_do_Dia'] = df['Tempo_Inicial'].dt.hour
    df['Dia_da_Semana'] = df['Tempo_Inicial'].dt.dayofweek
    df['Mes'] = df['Tempo_Inicial'].dt.month
    
    df['Horario_Pico'] = df['Hora_do_Dia'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 18) else 0)
    
    print("Criando zonas de risco espaciais usando MiniBatchKMeans...")
    coords = df[['Latitude_Inicial', 'Longitude_Inicial']].fillna(df[['Latitude_Inicial', 'Longitude_Inicial']].mean())
    kmeans = MiniBatchKMeans(n_clusters=20, random_state=42, batch_size=5000)
    df['Cluster_Espacial'] = kmeans.fit_predict(coords)
    
    df = df.drop(columns=['Tempo_Inicial'])
    return df

def handle_missing_data(df):
    print("\n--- 3. Tratamento de Dados Ausentes (Mediana e Moda) ---")
    cont_cols = ['Temperatura_F', 'Sensacao_Termica_F', 'Umidade_Percentual', 'Pressao_Polegadas', 'Visibilidade_Milhas', 'Velocidade_Vento_Mph', 'Precipitacao_Polegadas', 'Latitude_Inicial', 'Longitude_Inicial']
    cat_cols = ['Nascer_Por_Sol']
    
    for col in cont_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            
    return df

def detect_outliers_hybrid(df):
    print("\n--- 4. Detecção de Outliers (Mahalanobis) ---")
    num_cols = ['Temperatura_F', 'Umidade_Percentual', 'Pressao_Polegadas', 'Visibilidade_Milhas', 'Velocidade_Vento_Mph']
    x = df[num_cols].to_numpy()
    cov_matrix = np.cov(x, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    mean_val = np.mean(x, axis=0)
    diff = x - mean_val
    md = np.einsum('ij,jk,ik->i', diff, inv_cov_matrix, diff)
    
    df['P_Value'] = 1 - chi2.cdf(md, len(num_cols))
    outliers = df[df['P_Value'] < 0.001].index
    print(f"Detectados por Mahalanobis: {len(outliers)}")
    
    df_clean = df.drop(index=outliers).copy()
    if 'P_Value' in df_clean.columns:
        df_clean = df_clean.drop(columns=['P_Value'])
        
    return df_clean

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(project_root, "dataset")
    
    df = load_and_sample_data(folder, sample_size=None)
    df = map_target_to_3classes(df)
    df = feature_engineering(df)
    df = handle_missing_data(df)
    df = detect_outliers_hybrid(df)
    
    output_file = os.path.join(folder, "dataset_amostra_limpa_avancado_3classes.parquet")
    df.to_parquet(output_file, index=False)
    print(f"\nConcluído! Dataset de 3 classes limpo salvo em: {output_file}")
