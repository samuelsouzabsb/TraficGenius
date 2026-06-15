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
    num_cols = ['Temperatura_F', 'Sensacao_Termica_F', 'Umidade_Percentual', 'Pressao_Polegadas', 'Visibilidade_Milhas', 'Velocidade_Vento_Mph', 'Precipitacao_Polegadas']
    
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
    group_2 = df[df['Severidade'] == 2]
    group_3 = df[df['Severidade'] == 3]
    
    for col in num_cols:
        # Levene's Test (H0: As variâncias populacionais entre os grupos são iguais)
        stat, p = levene(group_2[col], group_3[col])
        is_homosc = "Sim" if p > 0.05 else "Nao"
        print(f" - {col[:15]:15}: P-Valor={p:.4f} | Homocedastico? {is_homosc}")

# Configura um estilo moderno de fundo escuro para a identidade visual dos gráficos do projeto
def apply_chart_style():
    plt.style.use('dark_background')
    plt.rcParams['figure.facecolor'] = '#121214'
    plt.rcParams['axes.facecolor'] = '#1a1a1e'
    plt.rcParams['grid.color'] = '#2d2d34'
    plt.rcParams['font.sans-serif'] = 'sans-serif'
    plt.rcParams['font.family'] = 'sans-serif'

def plot_normality_distributions(df, output_dir):
    """
    Gera um gráfico comparativo de subplots mostrando as curvas de distribuição empíricas (KDE)
    das variáveis meteorológicas vs curvas normais gaussianas ideais de referência.
    """
    apply_chart_style()
    print("Gerando gráfico comparativo de normalidade (KDE vs Gaussiana)...")
    num_cols = ['Temperatura_F', 'Umidade_Percentual', 'Pressao_Polegadas', 'Visibilidade_Milhas']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Amostra controlada para renderização suave e ágil
    df_sample = df.sample(n=10000, random_state=42)
    
    for idx, col in enumerate(num_cols):
        ax = axes[idx]
        data = df_sample[col].dropna()
        
        # Plota KDE real em rosa neon
        sns.kdeplot(data, ax=ax, fill=True, color='#ff007f', alpha=0.3, label='Distribuição Real', linewidth=2)
        
        # Plota curva Gaussiana teórica ideal em ciano neon
        mean, std = data.mean(), data.std()
        x_axis = np.linspace(data.min(), data.max(), 100)
        ax.plot(x_axis, norm.pdf(x_axis, mean, std), color='#00f0ff', linestyle='--', linewidth=2, label='Distribuição Normal Teórica')
        
        ax.set_title(f"Aderência à Normalidade: {col}", color='#00f0ff', fontsize=12, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("Densidade", color='#e2e2e9')
        ax.tick_params(colors='#e2e2e9')
        ax.legend(facecolor='#1a1a1e', edgecolor='#2d2d34', fontsize=9, labelcolor='#e2e2e9')
        
    plt.tight_layout()
    out_path = os.path.join(output_dir, "distribuicao_normalidade.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#121214')
    plt.close()
    print(f"Gráfico de normalidade gerado em: {out_path}")

def check_and_plot_correlation(df, output_dir):
    """
    Calcula a matriz de correlação linear de Pearson entre as variáveis numéricas
    e gera um mapa de calor (heatmap) elegante em fundo escuro com tema neon.
    """
    apply_chart_style()
    print("\n--- 2. Matriz de Correlação (Pearson) ---")
    num_cols = ['Temperatura_F', 'Sensacao_Termica_F', 'Umidade_Percentual', 'Pressao_Polegadas', 'Visibilidade_Milhas', 'Velocidade_Vento_Mph', 'Precipitacao_Polegadas', 'Distancia_Milhas', 'Hora_do_Dia', 'Mes', 'Latitude_Inicial']
    
    # Filtra as colunas realmente presentes na base
    existing_cols = [c for c in num_cols if c in df.columns]
    corr_matrix = df[existing_cols].corr(method='pearson')
    
    plt.figure(figsize=(12, 10))
    
    # Paleta divergente translúcida estilizada (Ciano a Rosa neon)
    cmap = sns.diverging_palette(180, 340, as_cmap=True)
    
    sns.heatmap(corr_matrix, annot=True, cmap=cmap, fmt=".2f", vmin=-1, vmax=1,
                cbar_kws={'shrink': 0.8}, linewidths=0.5, linecolor='#121214')
    
    plt.title("Matriz de Correlação (Pearson) - Features Contínuas", fontsize=16, color='#00f0ff', pad=20)
    plt.xticks(rotation=45, ha='right', color='#e2e2e9')
    plt.yticks(color='#e2e2e9')
    
    out_path = os.path.join(output_dir, "matriz_correlacao.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#121214')
    plt.close()
    print(f"Gráfico de correlação salvo em: {out_path}")

def check_multicollinearity(df, output_dir):
    """
    Analisa recursivamente a presença de colinearidade (multicolinearidade) severa
    usando o Fator de Inflação de Variância (VIF - Variance Inflation Factor).
    Gera um gráfico descritivo da situação de colinearidade inicial e
    remove as variáveis com VIF superior a 10 de forma sequencial (Stepwise reduction).
    """
    apply_chart_style()
    print("\n--- 3. Analise Iterativa de Multicolinearidade (VIF) ---")
    num_cols = ['Temperatura_F', 'Sensacao_Termica_F', 'Umidade_Percentual', 'Pressao_Polegadas', 'Visibilidade_Milhas', 'Velocidade_Vento_Mph', 'Precipitacao_Polegadas']
    
    # Amostra para acelerar o cálculo do VIF em grandes bases (estatisticamente idêntico)
    if len(df) > 100000:
        X = df[num_cols].dropna().sample(n=100000, random_state=42)
    else:
        X = df[num_cols].dropna()
    
    # Calcula os valores de VIF Iniciais para o relatório gráfico solicitado
    vif_initial = []
    for col in X.columns:
        y_vif = X[col]
        X_vif = X.drop(columns=[col])
        model = LinearRegression()
        model.fit(X_vif, y_vif)
        r_sq = model.score(X_vif, y_vif)
        vif = 1 / (1 - r_sq) if r_sq < 1 else float('inf')
        vif_initial.append((col, vif))
        
    vif_initial_df = pd.DataFrame(vif_initial, columns=['feature', 'VIF']).sort_values(by="VIF", ascending=True)
    
    # Plota a representação visual do VIF Inicial
    plt.figure(figsize=(10, 6))
    colors = ['#ff0055' if v > 10 else '#00f0ff' for v in vif_initial_df['VIF']]
    bars = plt.barh(vif_initial_df['feature'], vif_initial_df['VIF'], color=colors, height=0.6)
    
    # Adiciona linha guia crítica
    plt.axvline(x=10, color='#ffcc00', linestyle='--', linewidth=1.5, label='Limite Crítico (VIF = 10)')
    
    plt.title("Fator de Inflação de Variância (VIF) - Identificação de Colinearidade", fontsize=14, color='#00f0ff', pad=15)
    plt.xlabel("Valor do VIF", fontsize=12, color='#e2e2e9')
    plt.ylabel("Variáveis Numéricas", fontsize=12, color='#e2e2e9')
    plt.xticks(color='#e2e2e9')
    plt.yticks(color='#e2e2e9')
    plt.legend(facecolor='#1a1a1e', edgecolor='#2d2d34', labelcolor='#e2e2e9')
    
    # Rótulos nas barras
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                 f'{width:.2f}', 
                 va='center', ha='left', fontsize=10, color='#e2e2e9')
                 
    out_path = os.path.join(output_dir, "vif_multicolinearidade.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#121214')
    plt.close()
    print(f"Gráfico de VIF gerado em: {out_path}")
    
    # Executa o loop stepwise para remoção
    while True:
        vif_list = []
        for col in X.columns:
            y_vif = X[col]
            X_vif = X.drop(columns=[col])
            
            # Ajusta uma regressão linear comum para estimar a feature atual
            model = LinearRegression()
            model.fit(X_vif, y_vif)
            r_sq = model.score(X_vif, y_vif)  # Coeficiente de determinação (R²)
            
            vif = 1 / (1 - r_sq) if r_sq < 1 else float('inf')
            vif_list.append((col, vif))
            
        vif_df = pd.DataFrame(vif_list, columns=['feature', 'VIF']).sort_values(by="VIF", ascending=False)
        
        highest_vif_feature = vif_df.iloc[0]['feature']
        highest_vif_value = vif_df.iloc[0]['VIF']
        
        if highest_vif_value > 10:
            print(f"[REMOVIDO] {highest_vif_feature} devido a VIF crítico: {highest_vif_value:.2f}")
            X = X.drop(columns=[highest_vif_feature])
        else:
            print("\nVIF Final Seguro:")
            print(vif_df.to_string(index=False))
            break
            
    return X.columns.tolist()

if __name__ == "__main__":
    print("Iniciando Fase 2 (Refatorada)...")
    project_root = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(project_root, "dataset")
    file_path = os.path.join(folder_path, "dataset_amostra_limpa_avancado.parquet")
    
    print("Lendo parquet...")
    df = pd.read_parquet(file_path)
    print(f"Dataset carregado. Shape: {df.shape}")
    
    # Executa os testes de normalidade e homocedasticidade
    test_normality_homoscedasticity(df)
    
    # Gera o gráfico comparativo de aderência à normalidade
    plot_normality_distributions(df, folder_path)
    
    # Plota e salva a matriz de correlação de Pearson
    check_and_plot_correlation(df, folder_path)
    
    # Filtra atributos numéricos correlacionados recursivamente
    safe_features = check_multicollinearity(df, folder_path)
    print("\nFeatures Numéricas recomendadas (livres de colinearidade grave):")
    print(safe_features)
    
    print("Fase 2 finalizada!")
