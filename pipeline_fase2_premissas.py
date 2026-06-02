# -*- coding: utf-8 -*-
"""
Pipeline Fase 2: Testes de Premissas Estatísticas e Multicolinearidade
Este script valida os pressupostos teóricos exigidos por modelos de regressão e classificação tradicional.
O objetivo é garantir que os atributos numéricos tenham suas correlações e distribuições mapeadas,
eliminando a redundância de variáveis preditoras antes da fase de treinamento dos modelos.

Dicas de Inglês (English Tips):
- 'Normality' significa normalidade (adesão à distribuição normal).
- 'Homoscedasticity' refere-se à homocedasticidade (variância constante dos resíduos/variáveis entre grupos).
- 'Multicollinearity' significa multicolinearidade (quando variáveis explicativas são altamente correlacionadas entre si).
- 'VIF' (Variance Inflation Factor) significa Fator de Inflação da Variância.
- 'P-value' é o p-valor (probabilidade de observar o resultado assumindo a hipótese nula como verdadeira).
- 'Pearson correlation matrix' refere-se à matriz de correlação linear de Pearson.
"""

import pandas as pd
import numpy as np
from scipy.stats import shapiro, levene, kstest, norm
from sklearn.linear_model import LinearRegression
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ignora warnings para manter o console limpo
warnings.filterwarnings('ignore')

def test_normality_homoscedasticity(df):
    """
    Testa a aderência à distribuição normal das variáveis numéricas contínuas (Normalidade)
    e a igualdade de variâncias entre diferentes níveis de severidade (Homocedasticidade).
    
    Parâmetros (Parameters):
    - df (pd.DataFrame): Dataset com dados limpos.
    """
    print("--- 1. Testes de Normalidade Avançados e Homocedasticidade ---")
    num_cols = ['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']
    
    # 1.1 Amostragem Restrita
    # O teste de Shapiro-Wilk possui um limite computacional severo na biblioteca SciPy (N <= 5000).
    # Extraímos uma sub-amostra de 4.500 registros para garantir a execução bem-sucedida do teste.
    df_sample = df.sample(n=4500, random_state=42)
    
    print(f"Testando Normalidade (Shapiro-Wilk e Kolmogorov-Smirnov):")
    for col in num_cols:
        # Shapiro-Wilk Test (H0: Os dados seguem uma distribuição normal)
        stat_s, p_s = shapiro(df_sample[col])
        
        # Kolmogorov-Smirnov Test (KS)
        # Compara a distribuição empírica amostral com uma distribuição normal teórica ideal
        # parametrizada com a média e desvio padrão calculados diretamente dos dados amostrais.
        mean, std = df_sample[col].mean(), df_sample[col].std()
        stat_k, p_k = kstest(df_sample[col], 'norm', args=(mean, std))
        
        # Se p-valor > 0.05, não rejeitamos a hipótese nula H0, logo, consideramos normal.
        is_normal_s = "Sim" if p_s > 0.05 else "Não"
        is_normal_k = "Sim" if p_k > 0.05 else "Não"
        
        print(f" - {col[:15]:15}: Shapiro P={p_s:.4f} ({is_normal_s}) | K-S P={p_k:.4f} ({is_normal_k})")
        
    # 1.2 Teste de Homocedasticidade de Levene
    # Verifica se a variância estatística de cada variável contínua é estável/homogênea
    # entre os registros de acidentes moderados (Severidade 2) e graves (Severidade 3).
    print("\nTestando Homocedasticidade (Teste de Levene entre Severidade 2 e 3):")
    group_2 = df[df['Severity'] == 2]
    group_3 = df[df['Severity'] == 3]
    
    for col in num_cols:
        # Levene's Test (H0: As variâncias populacionais entre os grupos são iguais)
        stat, p = levene(group_2[col], group_3[col])
        is_homosc = "Sim" if p > 0.05 else "Nao"
        print(f" - {col[:15]:15}: P-Valor={p:.4f} | Homocedastico? {is_homosc}")

def check_and_plot_correlation(df, output_dir):
    """
    Calcula a matriz de correlação linear de Pearson entre as variáveis numéricas
    e gera um mapa de calor (heatmap) para representação visual das correlações.
    
    Parâmetros (Parameters):
    - df (pd.DataFrame): Dataset com dados limpos.
    - output_dir (str): Diretório onde a imagem gerada será gravada.
    """
    print("\n--- 2. Matriz de Correlação (Pearson) ---")
    num_cols = ['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 'Distance(mi)', 'Hora_do_Dia', 'Mes', 'Start_Lat']
    
    # Gera a matriz de coeficientes lineares (varia de -1 a +1)
    corr_matrix = df[num_cols].corr(method='pearson')
    
    # Configura e plota o mapa de calor com paleta divergente 'coolwarm'
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("Matriz de Correlação (Pearson) - Features Contínuas")
    
    # Salva o arquivo de imagem no diretório especificado
    out_path = os.path.join(output_dir, "matriz_correlacao.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico de correlação salvo em: {out_path}")

def check_multicollinearity(df):
    """
    Analisa recursivamente a presença de colinearidade (multicolinearidade) severa
    usando o Fator de Inflação de Variância (VIF - Variance Inflation Factor).
    Remove as variáveis com VIF superior a 10 de forma sequencial (Stepwise reduction)
    até que reste apenas um conjunto linearmente estável de variáveis preditoras.
    
    Parâmetros (Parameters):
    - df (pd.DataFrame): Dataset com dados limpos.
    
    Retorno (Returns):
    - safe_features (list): Lista de variáveis cujos níveis de VIF estão dentro do limite aceitável.
    """
    print("\n--- 3. Analise Iterativa de Multicolinearidade (VIF) ---")
    num_cols = ['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']
    
    # Cria uma cópia do conjunto numérico ignorando qualquer linha com nulo (dropna)
    X = df[num_cols].dropna()
    
    while True:
        vif_list = []
        # Para cada coluna na matriz X atual, calcula seu VIF em relação às demais
        for col in X.columns:
            y_vif = X[col]
            X_vif = X.drop(columns=[col])
            
            # Ajusta uma regressão linear comum para estimar a feature atual
            model = LinearRegression()
            model.fit(X_vif, y_vif)
            r_sq = model.score(X_vif, y_vif)  # Coeficiente de determinação (R²)
            
            # VIF = 1 / (1 - R²). Se R² se aproxima de 1, o VIF tende ao infinito.
            vif = 1 / (1 - r_sq) if r_sq < 1 else float('inf')
            vif_list.append((col, vif))
            
        # Ordena a lista de VIF de forma decrescente
        vif_df = pd.DataFrame(vif_list, columns=['feature', 'VIF']).sort_values(by="VIF", ascending=False)
        
        highest_vif_feature = vif_df.iloc[0]['feature']
        highest_vif_value = vif_df.iloc[0]['VIF']
        
        # O limiar (threshold) de VIF > 10 é uma regra empírica do mercado (rule of thumb)
        # indicando forte colinearidade multivariada.
        if highest_vif_value > 10:
            print(f"[REMOVIDO] {highest_vif_feature} devido a VIF crítico: {highest_vif_value:.2f}")
            # Remove apenas a pior variável e reavalia a matriz no próximo passo do loop (stepwise)
            X = X.drop(columns=[highest_vif_feature])
        else:
            # Todas as variáveis restantes possuem VIF estável <= 10
            print("\nVIF Final Seguro:")
            print(vif_df.to_string(index=False))
            break
            
    return X.columns.tolist()

if __name__ == "__main__":
    print("Iniciando Fase 2 (Refatorada)...")
    folder_path = r"c:\Users\samuelbarroso\Documents\Desenvolvimento\TraficGenius\dataset"
    file_path = os.path.join(folder_path, "dataset_amostra_limpa_avancado.parquet")
    
    print("Lendo parquet...")
    df = pd.read_parquet(file_path)
    print(f"Dataset carregado. Shape: {df.shape}")
    
    # Executa os testes de normalidade e homocedasticidade
    test_normality_homoscedasticity(df)
    
    # Plota e salva a matriz de correlação de Pearson
    check_and_plot_correlation(df, folder_path)
    
    # Filtra atributos numéricos correlacionados recursivamente
    safe_features = check_multicollinearity(df)
    print("\nFeatures Numéricas recomendadas (livres de colinearidade grave):")
    print(safe_features)
    
    print("Fase 2 finalizada!")
