# -*- coding: utf-8 -*-
"""
Pipeline Fase 1: Análise Exploratória de Dados (EDA), Feature Engineering e Limpeza Avançada
Este script implementa o primeiro estágio do pipeline de Machine Learning do TraficGenius.
Foca em eficiência de memória para lidar com conjuntos de dados volumosos de acidentes, 
tratamento estatístico avançado de dados faltantes (imputação múltipla) e filtragem de outliers.

Dicas de Inglês (English Tips):
- 'EDA' (Exploratory Data Analysis) significa Análise Exploratória de Dados.
- 'Feature Engineering' é a Engenharia de Recursos/Atributos (criação de novas variáveis para o modelo).
- 'Impute/Imputation' significa imputar ou preencher valores nulos (dados ausentes).
- 'MICE' (Multivariate Imputation by Chained Equations) é a técnica de Imputação Múltipla via Equações Encadeadas.
- 'Outliers' são valores discrepantes ou fora do padrão de comportamento da massa de dados.
- 'Rush Hour' refere-se ao Horário de Pico no trânsito.
"""

import pandas as pd
import numpy as np
import glob
import os
import warnings
from scipy.stats import chi2
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.cluster import MiniBatchKMeans

# Ignora avisos (warnings) do Python/scikit-learn para manter a saída de log limpa
warnings.filterwarnings('ignore')

def load_and_sample_data(folder_path, sample_size=200000):
    """
    Carrega os dados de forma otimizada para a memória RAM (Memory-Efficient loading)
    e extrai uma amostra aleatória estratificada/controlada baseada nos row groups do Parquet.
    
    Parâmetros (Parameters):
    - folder_path (str): Diretório contendo as bases de dados (.parquet).
    - sample_size (int): Quantidade máxima de linhas a serem selecionadas para amostragem.
    
    Retorno (Returns):
    - df_sample (pd.DataFrame): DataFrame do Pandas contendo a amostra selecionada.
    """
    import pyarrow.parquet as pq
    import random
    
    print("--- 1. Carregamento e Amostragem (Memory Efficient) ---")
    consolidated_file = os.path.join(folder_path, "dataset_consolidado.parquet")
    
    # Lista restrita de colunas de interesse para economizar memória durante a leitura (Select cols projection)
    usecols = [
        'Severity', 'Start_Time', 'Start_Lat', 'Start_Lng', 'Distance(mi)',
        'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 
        'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)',
        'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit',
        'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal',
        'Sunrise_Sunset'
    ]
    
    row_groups_info = []  # Lista para mapear onde cada bloco (row group) está
    
    # 1.1 Mapeia as posições dos blocos de dados nos arquivos disponíveis
    if os.path.exists(consolidated_file):
        print(f"Lendo base consolidada: {os.path.basename(consolidated_file)}")
        pf = pq.ParquetFile(consolidated_file)
        # Registra o arquivo consolidado e os índices de seus row groups
        for i in range(pf.num_row_groups):
            row_groups_info.append((consolidated_file, i))
    else:
        print("Base consolidada não encontrada. Buscando arquivos fragmentados...")
        files = glob.glob(os.path.join(folder_path, "train-*.parquet"))
        print(f"Encontrados {len(files)} arquivos parquet.")
        for f in files:
            pf = pq.ParquetFile(f)
            # Registra cada arquivo fragmentado e seus respectivos row groups
            for i in range(pf.num_row_groups):
                row_groups_info.append((f, i))
            
    print(f"Total de row_groups disponíveis: {len(row_groups_info)}")
    
    # 1.2 Embaralha as referências dos row groups para garantir que a amostragem seja aleatória e imparcial (unbiased sample)
    random.seed(42)
    random.shuffle(row_groups_info)
    
    dfs = []
    current_size = 0
    
    # 1.3 Lê sequencialmente apenas os blocos sorteados até alcançar o tamanho de amostra definido (sample_size)
    for f, i in row_groups_info:
        pf = pq.ParquetFile(f)
        # Lê apenas as colunas projetadas ('usecols') do row group sorteado
        df_part = pf.read_row_group(i, columns=usecols).to_pandas()
        dfs.append(df_part)
        current_size += len(df_part)
        # Critério de parada rápido para evitar carregar blocos desnecessários
        if current_size >= sample_size:
            break
            
    # Concatena os pedaços em um DataFrame único
    df_sample = pd.concat(dfs, ignore_index=True)
    
    # Ajusta para o tamanho exato solicitado caso a soma dos blocos passe ligeiramente de 'sample_size'
    if len(df_sample) > sample_size:
        df_sample = df_sample.sample(n=sample_size, random_state=42)
        
    print(f"Tamanho da amostra extraída: {len(df_sample)}")
    return df_sample

def feature_engineering(df):
    """
    Realiza Engenharia de Atributos (Feature Engineering) para extrair padrões latentes nos dados.
    Transforma data/hora e executa agrupamento geográfico por proximidade (Clustering).
    
    Parâmetros (Parameters):
    - df (pd.DataFrame): Conjunto de dados original.
    
    Retorno (Returns):
    - df (pd.DataFrame): DataFrame atualizado com as novas colunas.
    """
    print("\n--- 2. Feature Engineering Avançado ---")
    
    # 2.1 Extração de Informações Temporais (Temporal Feature Extraction)
    print("Extraindo features temporais...")
    # 'format=mixed' permite converter com sucesso mesmo que as strings de data tenham precisões de milissegundos variáveis
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='mixed')
    df['Hora_do_Dia'] = df['Start_Time'].dt.hour
    df['Dia_da_Semana'] = df['Start_Time'].dt.dayofweek
    df['Mes'] = df['Start_Time'].dt.month
    
    # Cria flag binária identificando Horário de Pico (Rush Hour) - Manhã (7-9h) ou Tarde (16-18h)
    df['Horario_Pico'] = df['Hora_do_Dia'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 18) else 0)
    
    # 2.2 Clustering Espacial (Zonas de Risco / Spatial Clustering)
    # Utilizamos o MiniBatchKMeans do scikit-learn pois é otimizado para grandes bases de dados,
    # processando as coordenadas geográficas em sub-lotes (mini-batches) de forma extremamente rápida.
    print("Criando zonas de risco espaciais usando MiniBatchKMeans...")
    # Preenche valores geográficos faltantes temporariamente com a média para evitar erros de execução no K-Means
    coords = df[['Start_Lat', 'Start_Lng']].fillna(df[['Start_Lat', 'Start_Lng']].mean())
    # Cria 20 grupos (clusters) espaciais que representarão zonas com perfis geográficos similares
    kmeans = MiniBatchKMeans(n_clusters=20, random_state=42, batch_size=5000)
    df['Cluster_Espacial'] = kmeans.fit_predict(coords)
    
    # Remove a coluna original de data/hora 'Start_Time' para otimizar o uso de RAM,
    # uma vez que já extraímos todas as informações numéricas relevantes dela.
    df = df.drop(columns=['Start_Time'])
    
    return df

def handle_missing_data(df):
    """
    Trata dados ausentes (missing values) usando imputação múltipla baseada em regressões lineares
    (MICE - Multivariate Imputation by Chained Equations) e imputação categórica pela moda.
    
    Parâmetros (Parameters):
    - df (pd.DataFrame): Dataset com presença de valores nulos.
    
    Retorno (Returns):
    - df (pd.DataFrame): Dataset tratado com dados completamente preenchidos.
    """
    print("\n--- 3. Tratamento Avançado de Dados Ausentes (Iterative Imputer / MICE) ---")
    missing_pct = df.isnull().mean() * 100
    print("Percentual de dados ausentes por coluna (>0%):")
    print(missing_pct[missing_pct > 0].sort_values(ascending=False))
    
    # Variáveis numéricas contínuas elegíveis para a imputação avançada por MICE
    cont_cols = ['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 'Start_Lat', 'Start_Lng']
    # Variáveis categóricas tratadas de forma simples por representatividade de classe
    cat_cols = ['Sunrise_Sunset']
    
    print("Iniciando MICE nas variáveis contínuas (isso pode levar alguns segundos)...")
    # IterativeImputer estima recursivamente cada feature contínua em função das demais.
    # Configuramos um limite de 3 iterações (max_iter=3) e dependência de 4 atributos mais próximos
    # (n_nearest_features=4) para manter um equilíbrio ideal entre velocidade e precisão.
    imputer = IterativeImputer(max_iter=3, random_state=42, n_nearest_features=4)
    df[cont_cols] = imputer.fit_transform(df[cont_cols])
            
    # Imputação de Variáveis Categóricas usando a Moda (valor mais comum/frequente)
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    print("Valores ausentes após imputação:", df.isnull().sum().sum())
    return df

def detect_outliers_hybrid(df):
    """
    Identifica e remove valores discrepantes (outliers) no espaço multivariado
    utilizando uma abordagem híbrida de filtragem:
    1. Distância de Mahalanobis: Abordagem paramétrica linear que considera a covariância dos dados.
    2. Isolation Forest: Abordagem não-linear de aprendizado de máquina baseada em árvores de decisão.
    
    Parâmetros (Parameters):
    - df (pd.DataFrame): Dataset original.
    
    Retorno (Returns):
    - df_clean (pd.DataFrame): Dataset livre dos outliers identificados em ambas as abordagens.
    """
    print("\n--- 4. Detecção de Outliers Híbrida (Mahalanobis + Isolation Forest) ---")
    num_cols = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)']
    
    # --- Etapa A: Distância de Mahalanobis (Linear Outlier Detection) ---
    print("Etapa A: Mahalanobis (Linear)...")
    x = df[num_cols].to_numpy()
    cov_matrix = np.cov(x, rowvar=False)  # Matriz de covariância
    inv_cov_matrix = np.linalg.inv(cov_matrix)  # Inversa da matriz de covariância
    mean_val = np.mean(x, axis=0)  # Vetor de médias
    diff = x - mean_val
    # Multiplicação matricial acelerada (Einstein summation convention) para obter a distância de Mahalanobis ao quadrado
    md = np.einsum('ij,jk,ik->i', diff, inv_cov_matrix, diff)
    
    # Compara a distância quadrada de Mahalanobis com o limite crítico da distribuição Qui-Quadrado (chi2)
    df['P_Value'] = 1 - chi2.cdf(md, len(num_cols))
    # Registra índices cujo p-valor seja menor que 0.001 (significância alta de outlier)
    outliers_mah = df[df['P_Value'] < 0.001].index
    print(f"Detectados por Mahalanobis: {len(outliers_mah)}")
    
    # --- Etapa B: Isolation Forest (Non-Linear Outlier Detection) ---
    print("Etapa B: Isolation Forest (Não-Linear)...")
    # Instancia a floresta com contaminação fixada em 1% (contamination=0.01) e utilizando todos os núcleos de CPU (n_jobs=-1)
    iso = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
    preds = iso.fit_predict(df[num_cols])
    # Os outliers são representados pelo valor predito de -1
    outliers_iso = df.index[preds == -1]
    print(f"Detectados por Isolation Forest: {len(outliers_iso)}")
    
    # União lógica dos outliers (remove pontos identificados por qualquer uma das duas técnicas)
    all_outliers = list(set(outliers_mah).union(set(outliers_iso)))
    print(f"Total combinado de Outliers a remover: {len(all_outliers)}")
    
    # Remove as linhas discrepantes do DataFrame e gera uma cópia em memória
    df_clean = df.drop(index=all_outliers).copy()
    if 'P_Value' in df_clean.columns:
        df_clean = df_clean.drop(columns=['P_Value'])
        
    print(f"Tamanho do dataset após limpeza: {len(df_clean)}")
    
    return df_clean

if __name__ == "__main__":
    # Caminho do diretório de dados
    folder = r"c:\Users\samuelbarroso\Documents\Desenvolvimento\TraficGenius\dataset"
    
    # 1. Executa a amostragem eficiente em memória (200.000 registros para EDA)
    df = load_and_sample_data(folder, sample_size=200000)
    
    # 2. Executa a criação de variáveis temporais e espaciais
    df = feature_engineering(df)
    
    # 3. Executa a imputação inteligente de dados ausentes (MICE)
    df = handle_missing_data(df)
    
    # 4. Executa a remoção híbrida de outliers (Mahalanobis + Isolation Forest)
    df = detect_outliers_hybrid(df)
    
    # 5. Salva o conjunto resultante limpo de alta qualidade no disco
    output_file = os.path.join(folder, "dataset_amostra_limpa_avancado.parquet")
    df.to_parquet(output_file, index=False)
    print(f"\nConcluído! Dataset avançado salvo em: {output_file}")
