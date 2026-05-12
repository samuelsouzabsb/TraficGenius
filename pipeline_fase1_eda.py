print("Importando pandas...")
import pandas as pd
print("Importando numpy...")
import numpy as np
import glob
import os
print("Importando scipy...")
from scipy.stats import chi2
print("Imports concluídos.")

def load_and_sample_data(folder_path, sample_size=200000):
    import pyarrow.parquet as pq
    import random
    
    print("--- 1. Carregamento e Amostragem (Memory Efficient) ---")
    files = glob.glob(os.path.join(folder_path, "train-*.parquet"))
    print(f"Encontrados {len(files)} arquivos parquet.")
    
    usecols = [
        'Severity', 'Start_Time', 'Start_Lat', 'Start_Lng', 'Distance(mi)',
        'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 
        'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)',
        'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit',
        'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal',
        'Sunrise_Sunset'
    ]
    
    # Coletar informações sobre as row_groups
    row_groups_info = []
    for f in files:
        pf = pq.ParquetFile(f)
        for i in range(pf.num_row_groups):
            row_groups_info.append((f, i))
            
    print(f"Total de row_groups disponíveis: {len(row_groups_info)}")
    
    # Embaralhar as row_groups para garantir amostragem aleatória
    random.seed(42)
    random.shuffle(row_groups_info)
    
    dfs = []
    current_size = 0
    
    for f, i in row_groups_info:
        pf = pq.ParquetFile(f)
        # Lê apenas a row_group específica
        df_part = pf.read_row_group(i, columns=usecols).to_pandas()
        dfs.append(df_part)
        current_size += len(df_part)
        if current_size >= sample_size:
            break
            
    df_sample = pd.concat(dfs, ignore_index=True)
    # Se passou do tamanho, corta o excedente
    if len(df_sample) > sample_size:
        df_sample = df_sample.sample(n=sample_size, random_state=42)
        
    print(f"Tamanho da amostra extraída: {len(df_sample)}")
    print("Distribuição de Severidade na amostra:")
    print(df_sample['Severity'].value_counts(normalize=True))
    return df_sample

def handle_missing_data(df):
    print("\n--- 2. Tratamento de Dados Ausentes (Missing Data) ---")
    missing_pct = df.isnull().mean() * 100
    print("Percentual de dados ausentes por coluna (>0%):")
    print(missing_pct[missing_pct > 0].sort_values(ascending=False))
    
    # Estratégia: 
    # Variáveis contínuas -> Imputação pela Mediana (robusto a outliers)
    # Variáveis categóricas -> Imputação pela Moda
    
    cont_cols = ['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']
    cat_cols = ['Sunrise_Sunset']
    
    # Imputação Contínuas
    for col in cont_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    # Imputação Categóricas
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    print("Valores ausentes após imputação:", df.isnull().sum().sum())
    return df

def detect_outliers_mahalanobis(df):
    print("\n--- 3. Detecção de Outliers Multivariados (Mahalanobis) ---")
    # Selecionar variáveis numéricas contínuas para Mahalanobis
    num_cols = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)']
    
    # Certificar que não há NaN antes da matriz de covariância
    x = df[num_cols].to_numpy()
    
    # Calcula a matriz de covariância
    cov_matrix = np.cov(x, rowvar=False)
    # Inversa da matriz de covariância
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    # Media das colunas
    mean_val = np.mean(x, axis=0)
    
    # Distância de Mahalanobis para cada linha
    diff = x - mean_val
    # np.einsum é muito mais rápido e eficiente em memória para cálculo de Mahalanobis
    md = np.einsum('ij,jk,ik->i', diff, inv_cov_matrix, diff)
    
    # Adicionando ao DataFrame
    df['Mahalanobis'] = md
    
    # Cálculo do P-Valor usando a distribuição Qui-Quadrado (graus de liberdade = num de variáveis)
    df['P_Value'] = 1 - chi2.cdf(df['Mahalanobis'], len(num_cols))
    
    # Critério: P-Valor < 0.001 (0.1%) para ser considerado um outlier extremo multivariado
    outliers = df[df['P_Value'] < 0.001]
    print(f"Total de Outliers detectados: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    
    # Excluindo outliers
    df_clean = df[df['P_Value'] >= 0.001].copy()
    df_clean = df_clean.drop(columns=['Mahalanobis', 'P_Value'])
    print(f"Tamanho do dataset após exclusão: {len(df_clean)}")
    
    return df_clean

if __name__ == "__main__":
    folder = r"c:\Users\samuelbarroso\Documents\Desenvolvimento\TraficGenius\dataset"
    
    # 1. Amostragem
    df = load_and_sample_data(folder, sample_size=200000)
    
    # 2. Imputação (MAR / MCAR)
    # Optamos por imputação simples aqui por performance e robustez (mediana)
    df = handle_missing_data(df)
    
    # 3. Outliers (Mahalanobis)
    df = detect_outliers_mahalanobis(df)
    
    # 4. Salvar amostra limpa
    output_file = os.path.join(folder, "dataset_amostra_limpa.parquet")
    df.to_parquet(output_file, index=False)
    print(f"\nConcluído! Dataset limpo salvo em: {output_file}")
