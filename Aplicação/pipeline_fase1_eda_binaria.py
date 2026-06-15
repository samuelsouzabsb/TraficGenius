# -*- coding: utf-8 -*-
"""
Pipeline Fase 1: Análise Exploratória de Dados (EDA), Feature Engineering e Limpeza Avançada (Binário)
Atua como Cientista de Dados Sênior aplicando tratamento estatístico rigoroso.
Todas as colunas são traduzidas para o português do Brasil antes do processamento.
Otimizado para processar a base completa de 8,1 milhões de registros sem estouro de memória.
"""

import pandas as pd
import numpy as np
import os
import warnings
import gc
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Dicionário de tradução completo para o português do Brasil
COLUMNS_TRANSLATION = {
    'grau_severidade': 'grau_severidade',
    'latitude': 'latitude',
    'longitude': 'longitude',
    'pais': 'pais',
    'data_inversa': 'data_inversa',
    'horario': 'horario',
    'h3_9': 'h3_res9',
    'lat_centro_9': 'latitude_centro_res9',
    'lon_centro_9': 'longitude_centro_res9',
    'h3_10': 'h3_res10',
    'lat_centro_10': 'latitude_centro_res10',
    'lon_centro_10': 'longitude_centro_res10',
    'h3_11': 'h3_res11',
    'lat_centro_11': 'latitude_centro_res11',
    'lon_centro_11': 'longitude_centro_res11',
    'n_cruzamentos': 'quantidade_cruzamentos',
    'n_semaforos': 'quantidade_semaforos',
    'speed_mean': 'velocidade_media',
    'speed_min': 'velocidade_minima',
    'speed_max': 'velocidade_maxima',
    'lanes_mean': 'quantidade_faixas_media',
    'curv_accumulated': 'curvatura_acumulada',
    'curv_max_deviation': 'desvio_maximo_curvatura',
    'curv_sharp_count': 'quantidade_curvas_acentuadas',
    'n_rotatorias': 'quantidade_rotatorias',
    'n_pontes': 'quantidade_pontes',
    'bridge_length_m': 'comprimento_pontes_metros',
    'n_tuneis': 'quantidade_tuneis',
    'tunnel_length_m': 'comprimento_tuneis_metros',
    'road_length_m_h11': 'extensao_rodovia_metros_res11',
    'dominant_highway_h11': 'rodovia_dominante_res11',
    'n_postos': 'quantidade_postos_combustivel',
    'n_restaurantes': 'quantidade_restaurantes',
    'n_escolas': 'quantidade_escolas',
    'road_length_m_h10': 'extensao_rodovia_metros_res10',
    'dominant_highway_h10': 'rodovia_dominante_res10',
    'n_hospitais': 'quantidade_hospitais',
    'road_count_distinct': 'quantidade_rodovias_distintas',
    'total_sharp_curves': 'total_curvas_acentuadas',
    'place_count': 'quantidade_locais_interesse',
    'urban_area_m2': 'area_urbana_m2',
    'rural_area_m2': 'area_rural_m2',
    'road_length_m_h9': 'extensao_rodovia_metros_res9',
    'place_type': 'tipo_local',
    'dominant_highway_h9': 'rodovia_dominante_res9',
    'urban_ratio': 'proporcao_urbana',
    'temperature_c': 'temperatura_celsius',
    'dew_point_c': 'ponto_orvalho_celsius',
    'pressure_hpa': 'pressao_hpa',
    'wind_u': 'velocidade_vento_u',
    'wind_v': 'velocidade_vento_v',
    'cloud_cover': 'cobertura_nuvens_percentual',
    'precip_mm': 'precipitacao_milimetros'
}

def load_and_sample_data(file_path, sample_size=None):
    """
    Carrega o dataset completo, traduz todas as colunas para o português e
    gerencia o lixo de memória para evitar Out-of-Memory.
    """
    print(f"--- 1. Carregamento Completo e Tradução de Colunas ---")
    print(f"Lendo base consolidada: {os.path.basename(file_path)}")
    
    usecols = [
        'grau_severidade', 'data_inversa', 'horario', 'pais',
        'temperature_c', 'dew_point_c', 'pressure_hpa', 'wind_u', 'wind_v', 'cloud_cover', 'precip_mm'
    ]
    
    # Carrega todo o dataset em DataFrame
    df_full = pd.read_parquet(file_path, columns=usecols)
    print(f"Dataset original carregado. Shape: {df_full.shape}")
    
    # Traduz as colunas do dataset completo
    print("Traduzindo todas as colunas do dataset para o português...")
    df_full = df_full.rename(columns=COLUMNS_TRANSLATION)
    
    if sample_size is not None and sample_size < len(df_full):
        print(f"Extraindo amostra estratificada de {sample_size:,} registros...")
        _, df_sample = train_test_split(
            df_full, 
            test_size=sample_size / len(df_full), 
            stratify=df_full['grau_severidade'], 
            random_state=42
        )
        df_sample = df_sample.copy()
        print(f"Amostra gerada. Shape: {df_sample.shape}")
        del df_full
        gc.collect()
        return df_sample
    
    return df_full

def feature_engineering(df):
    """
    Decompõe variáveis temporais de forma otimizada para evitar alocação excessiva.
    """
    print(f"\n--- 2. Feature Engineering Temporária ---")
    
    # Converte data_inversa para datetime
    df['data_inversa'] = pd.to_datetime(df['data_inversa'])
    
    # Extração de componentes
    df['Hora_do_Dia'] = df['data_inversa'].dt.hour
    df['Dia_da_Semana'] = df['data_inversa'].dt.dayofweek
    df['Mes'] = df['data_inversa'].dt.month
    
    # Horário de pico (Rush Hour)
    df['Horario_Pico'] = df['Hora_do_Dia'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 18) else 0)
    
    print("Engenharia de recursos temporais concluída.")
    return df

def handle_missing_data(df):
    """
    Tratamento científico de missing data.
    """
    print(f"\n--- 3. Tratamento de Dados Ausentes ---")
    
    # 3.1 Drop de colunas com alta taxa de ausência (> 70%)
    drop_cols = ['tipo_local', 'velocidade_minima', 'velocidade_maxima', 'proporcao_urbana']
    print(f"Removendo variáveis com mais de 70% de dados nulos: {drop_cols}")
    df = df.drop(columns=drop_cols)
    
    # 3.2 Imputação com zero para variáveis de infraestrutura que implicam contagem
    zero_impute_cols = [
        'quantidade_cruzamentos', 'quantidade_semaforos', 'quantidade_rotatorias', 
        'quantidade_pontes', 'quantidade_tuneis', 'quantidade_postos_combustivel', 
        'quantidade_restaurantes', 'quantidade_escolas', 'quantidade_hospitais', 
        'quantidade_locais_interesse', 'quantidade_curvas_acentuadas', 'total_curvas_acentuadas', 
        'comprimento_pontes_metros', 'comprimento_tuneis_metros', 'area_urbana_m2', 
        'area_rural_m2', 'curvatura_acumulada', 'desvio_maximo_curvatura'
    ]
    print(f"Preenchendo nulos com 0 para variáveis de contagem e infraestrutura...")
    for col in zero_impute_cols:
        df[col] = df[col].fillna(0)
        
    # 3.3 Imputação com categoria especial para classes de rodovia
    cat_cols = ['rodovia_dominante_res11', 'rodovia_dominante_res10', 'rodovia_dominante_res9']
    print("Preenchendo nulos das rodovias dominantes com a categoria 'desconhecido'...")
    for col in cat_cols:
        df[col] = df[col].fillna('desconhecido')
        
    # 3.4 Imputação das extensões de rodovia e quantidade de vias distintas com a mediana global
    median_impute_cols = ['extensao_rodovia_metros_res11', 'extensao_rodovia_metros_res10', 'extensao_rodovia_metros_res9', 'quantidade_rodovias_distintas']
    print("Preenchendo comprimentos de pista nulos com a mediana global...")
    for col in median_impute_cols:
        df[col] = df[col].fillna(df[col].median())
        
    # 3.5 Imputação de velocidade média e faixas usando mediana condicional por tipo de via
    print("Imputando velocidade média e faixas usando a mediana condicional do tipo de via (rodovia_dominante_res11)...")
    speed_medians = df.groupby('rodovia_dominante_res11')['velocidade_media'].transform('median')
    df['velocidade_media'] = df['velocidade_media'].fillna(speed_medians)
    df['velocidade_media'] = df['velocidade_media'].fillna(df['velocidade_media'].median())
    
    lanes_medians = df.groupby('rodovia_dominante_res11')['quantidade_faixas_media'].transform('median')
    df['quantidade_faixas_media'] = df['quantidade_faixas_media'].fillna(lanes_medians)
    df['quantidade_faixas_media'] = df['quantidade_faixas_media'].fillna(df['quantidade_faixas_media'].median())
    
    # Verifica se restam dados nulos
    null_counts = df.isna().sum()
    null_remaining = null_counts[null_counts > 0]
    if len(null_remaining) > 0:
        print(f"AVISO: Ainda restam nulos em: {null_remaining.to_dict()}")
    else:
        print("Sucesso! Nenhum valor ausente restante no dataset.")
        
    return df

def detect_outliers_mahalanobis(df):
    """
    Remove outliers usando a Distância de Mahalanobis.
    Processado em blocos para evitar picos de uso de RAM no dataset completo de 8.1M.
    """
    print(f"\n--- 4. Detecção e Remoção Híbrida de Outliers (Mahalanobis - Otimizada) ---")
    
    mahalanobis_cols = [
        'velocidade_media', 'quantidade_faixas_media', 'curvatura_acumulada', 'desvio_maximo_curvatura',
        'extensao_rodovia_metros_res11', 'extensao_rodovia_metros_res10', 'extensao_rodovia_metros_res9',
        'temperatura_celsius', 'ponto_orvalho_celsius', 'pressao_hpa', 
        'velocidade_vento_u', 'velocidade_vento_v', 'cobertura_nuvens_percentual', 'precipitacao_milimetros'
    ]
    
    # Prepara matriz X e escala
    X = df[mahalanobis_cols].to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calcula covariância com regularização
    cov = np.cov(X_scaled, rowvar=False)
    cov += np.eye(cov.shape[0]) * 1e-6
    inv_cov = np.linalg.inv(cov)
    
    mean_val = np.mean(X_scaled, axis=0)
    diff = X_scaled - mean_val
    
    # Libera arrays brutos
    del X, X_scaled
    gc.collect()
    
    # Calcula a distância D^2 em blocos para economizar memória
    md = np.zeros(len(df))
    chunk_size = 1000000
    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))
        diff_chunk = diff[start:end]
        md[start:end] = np.einsum('ij,jk,ik->i', diff_chunk, inv_cov, diff_chunk)
        
    del diff
    gc.collect()
    
    # Determina P-Valores pela distribuição Qui-Quadrado
    p_values = 1 - chi2.cdf(md, df=len(mahalanobis_cols))
    
    # Outliers (p < 0.001)
    outliers_mask = p_values < 0.001
    num_outliers = outliers_mask.sum()
    
    print(f"Total de registros analisados: {len(df):,}")
    print(f"Outliers multivariados detectados (p < 0.001): {num_outliers:,} ({num_outliers/len(df)*100:.2f}%)")
    
    df_clean = df[~outliers_mask].copy()
    print(f"Tamanho do dataset limpo de outliers: {len(df_clean):,}")
    return df_clean

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(project_root, "dataset", "dados_unificados.parquet")
    output_file = os.path.join(project_root, "dataset", "dataset_amostra_limpa_binaria.parquet")
    
    # Carregamento do dataset unificado contendo os campos de ambos os países
    df = load_and_sample_data(input_file, sample_size=None)
    
    # Mapeia target para binário
    print("\nMapeando grau de severidade original para target binário...")
    df['severidade_binaria'] = df['grau_severidade'].map({1: 0, 2: 0, 3: 1, 4: 1})
    print("Frequência das classes no target binário:")
    print(df['severidade_binaria'].value_counts(normalize=True))
    
    # Remove coluna original
    df = df.drop(columns=['grau_severidade'])
    
    # Engenharia de atributos
    df = feature_engineering(df)
    
    # Tratamento de dados ausentes
    df = handle_missing_data(df)
    
    # Detecção de outliers
    df = detect_outliers_mahalanobis(df)
    
    # Criação de variáveis de interação física (Pontos 2 do plano de melhorias)
    print("\n--- 5. Criação de Variáveis de Interação Física ---")
    df['interacao_velocidade_faixas'] = df['velocidade_media'] * df['quantidade_faixas_media']
    df['interacao_chuva_curvas'] = df['precipitacao_milimetros'] * df['total_curvas_acentuadas']
    df['interacao_clima_curvatura'] = df['temperatura_celsius'] * df['curvatura_acumulada']
    print("Variáveis de interação criadas com sucesso.")
    
    # Salva dataset completo
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_parquet(output_file, index=False)
    print(f"\nSucesso! Dataset completo limpo binário salvo em: {output_file}")
