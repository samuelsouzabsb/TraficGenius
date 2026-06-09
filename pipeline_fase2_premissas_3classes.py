# -*- coding: utf-8 -*-
"""
Pipeline Fase 2: Testes de Premissas Estatísticas e Multicolinearidade (3 Classes)
"""

import pandas as pd
import numpy as np
from scipy.stats import shapiro, levene, kstest, norm
from sklearn.linear_model import LinearRegression
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os

warnings.filterwarnings('ignore')

def test_normality_homoscedasticity(df):
    print("--- 1. Testes de Normalidade e Homocedasticidade (3 Classes) ---")
    num_cols = ['Temperatura_F', 'Sensacao_Termica_F', 'Umidade_Percentual', 'Pressao_Polegadas', 'Visibilidade_Milhas', 'Velocidade_Vento_Mph', 'Precipitacao_Polegadas']
    
    df_sample = df.sample(n=4500, random_state=42)
    
    print(f"Testando Normalidade (Shapiro-Wilk e Kolmogorov-Smirnov):")
    for col in num_cols:
        stat_s, p_s = shapiro(df_sample[col])
        mean, std = df_sample[col].mean(), df_sample[col].std()
        stat_k, p_k = kstest(df_sample[col], 'norm', args=(mean, std))
        
        is_normal_s = "Sim" if p_s > 0.05 else "Não"
        is_normal_k = "Sim" if p_k > 0.05 else "Não"
        print(f" - {col[:15]:15}: Shapiro P={p_s:.4f} ({is_normal_s}) | K-S P={p_k:.4f} ({is_normal_k})")
        
    print("\nTestando Homocedasticidade (Teste de Levene entre Severidade 1 [Leve/Médio] e 2 [Grave]):")
    group_1 = df[df['Severidade'] == 1]
    group_2 = df[df['Severidade'] == 2]
    
    for col in num_cols:
        stat, p = levene(group_1[col], group_2[col])
        is_homosc = "Sim" if p > 0.05 else "Nao"
        print(f" - {col[:15]:15}: P-Valor={p:.4f} | Homocedastico? {is_homosc}")

def apply_chart_style():
    plt.style.use('dark_background')
    plt.rcParams['figure.facecolor'] = '#121214'
    plt.rcParams['axes.facecolor'] = '#1a1a1e'
    plt.rcParams['grid.color'] = '#2d2d34'
    plt.rcParams['font.sans-serif'] = 'sans-serif'
    plt.rcParams['font.family'] = 'sans-serif'

def plot_normality_distributions(df, output_dir):
    apply_chart_style()
    print("Gerando gráfico comparativo de normalidade (KDE vs Gaussiana - 3 Classes)...")
    num_cols = ['Temperatura_F', 'Umidade_Percentual', 'Pressao_Polegadas', 'Visibilidade_Milhas']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    df_sample = df.sample(n=10000, random_state=42)
    
    for idx, col in enumerate(num_cols):
        ax = axes[idx]
        data = df_sample[col].dropna()
        
        sns.kdeplot(data, ax=ax, fill=True, color='#ff007f', alpha=0.3, label='Distribuição Real', linewidth=2)
        mean, std = data.mean(), data.std()
        x_axis = np.linspace(data.min(), data.max(), 100)
        ax.plot(x_axis, norm.pdf(x_axis, mean, std), color='#00f0ff', linestyle='--', linewidth=2, label='Distribuição Normal Teórica')
        
        ax.set_title(f"Aderência à Normalidade (3C): {col}", color='#00f0ff', fontsize=12, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("Densidade", color='#e2e2e9')
        ax.tick_params(colors='#e2e2e9')
        ax.legend(facecolor='#1a1a1e', edgecolor='#2d2d34', fontsize=9, labelcolor='#e2e2e9')
        
    plt.tight_layout()
    out_path = os.path.join(output_dir, "distribuicao_normalidade_3classes.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#121214')
    plt.close()
    print(f"Gráfico de normalidade gerado em: {out_path}")

def check_and_plot_correlation(df, output_dir):
    apply_chart_style()
    print("\n--- 2. Matriz de Correlação (Pearson - 3 Classes) ---")
    num_cols = ['Temperatura_F', 'Sensacao_Termica_F', 'Umidade_Percentual', 'Pressao_Polegadas', 'Visibilidade_Milhas', 'Velocidade_Vento_Mph', 'Precipitacao_Polegadas', 'Distancia_Milhas', 'Hora_do_Dia', 'Mes', 'Latitude_Inicial']
    
    existing_cols = [c for c in num_cols if c in df.columns]
    corr_matrix = df[existing_cols].corr(method='pearson')
    
    plt.figure(figsize=(12, 10))
    cmap = sns.diverging_palette(180, 340, as_cmap=True)
    sns.heatmap(corr_matrix, annot=True, cmap=cmap, fmt=".2f", vmin=-1, vmax=1,
                cbar_kws={'shrink': 0.8}, linewidths=0.5, linecolor='#121214')
    
    plt.title("Matriz de Correlação (Pearson) - Features Contínuas (3 Classes)", fontsize=16, color='#00f0ff', pad=20)
    plt.xticks(rotation=45, ha='right', color='#e2e2e9')
    plt.yticks(color='#e2e2e9')
    
    out_path = os.path.join(output_dir, "matriz_correlacao_3classes.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#121214')
    plt.close()
    print(f"Gráfico de correlação salvo em: {out_path}")

def check_multicollinearity(df, output_dir):
    apply_chart_style()
    print("\n--- 3. Analise Iterativa de Multicolinearidade (VIF - 3 Classes) ---")
    num_cols = ['Temperatura_F', 'Sensacao_Termica_F', 'Umidade_Percentual', 'Pressao_Polegadas', 'Visibilidade_Milhas', 'Velocidade_Vento_Mph', 'Precipitacao_Polegadas']
    
    if len(df) > 100000:
        X = df[num_cols].dropna().sample(n=100000, random_state=42)
    else:
        X = df[num_cols].dropna()
    
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
    
    plt.figure(figsize=(10, 6))
    colors = ['#ff0055' if v > 10 else '#00f0ff' for v in vif_initial_df['VIF']]
    bars = plt.barh(vif_initial_df['feature'], vif_initial_df['VIF'], color=colors, height=0.6)
    plt.axvline(x=10, color='#ffcc00', linestyle='--', linewidth=1.5, label='Limite Crítico (VIF = 10)')
    
    plt.title("Fator de Inflação de Variância (VIF) - 3 Classes", fontsize=14, color='#00f0ff', pad=15)
    plt.xlabel("Valor do VIF", fontsize=12, color='#e2e2e9')
    plt.ylabel("Variáveis Numéricas", fontsize=12, color='#e2e2e9')
    plt.xticks(color='#e2e2e9')
    plt.yticks(color='#e2e2e9')
    plt.legend(facecolor='#1a1a1e', edgecolor='#2d2d34', labelcolor='#e2e2e9')
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                 f'{width:.2f}', 
                 va='center', ha='left', fontsize=10, color='#e2e2e9')
                 
    out_path = os.path.join(output_dir, "vif_multicolinearidade_3classes.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#121214')
    plt.close()
    print(f"Gráfico de VIF gerado em: {out_path}")
    
    while True:
        vif_list = []
        for col in X.columns:
            y_vif = X[col]
            X_vif = X.drop(columns=[col])
            
            model = LinearRegression()
            model.fit(X_vif, y_vif)
            r_sq = model.score(X_vif, y_vif)
            
            vif = 1 / (1 - r_sq) if r_sq < 1 else float('inf')
            vif_list.append((col, vif))
            
        vif_df = pd.DataFrame(vif_list, columns=['feature', 'VIF']).sort_values(by="VIF", ascending=False)
        highest_vif_feature = vif_df.iloc[0]['feature']
        highest_vif_value = vif_df.iloc[0]['VIF']
        
        if highest_vif_value > 10:
            print(f"[REMOVIDO] {highest_vif_feature} devido a VIF crítico: {highest_vif_value:.2f}")
            X = X.drop(columns=[highest_vif_feature])
        else:
            print("\nVIF Final Seguro (3 Classes):")
            print(vif_df.to_string(index=False))
            break
            
    return X.columns.tolist()

if __name__ == "__main__":
    print("Iniciando Fase 2 (3 Classes)...")
    project_root = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(project_root, "dataset")
    file_path = os.path.join(folder_path, "dataset_amostra_limpa_avancado_3classes.parquet")
    
    # Se o dataset de 3 classes ainda não tiver sido gerado pela Fase 1, lê o consolidado original e mapeia na hora
    if not os.path.exists(file_path):
        print(f"Aviso: {file_path} não encontrado. Por favor, execute pipeline_fase1_eda_3classes.py primeiro.")
        # Fallback para carregar a versão de 4 classes e mapear a severidade em memória
        fallback_path = os.path.join(folder_path, "dataset_amostra_limpa_avancado.parquet")
        print(f"Carregando fallback de 4 classes em {fallback_path}...")
        df = pd.read_parquet(fallback_path)
        df['Severidade'] = df['Severidade'].map({1: 1, 2: 1, 3: 2, 4: 3})
    else:
        print(f"Lendo parquet de 3 classes em {file_path}...")
        df = pd.read_parquet(file_path)
        
    print(f"Dataset carregado. Shape: {df.shape}")
    
    test_normality_homoscedasticity(df)
    plot_normality_distributions(df, folder_path)
    check_and_plot_correlation(df, folder_path)
    safe_features = check_multicollinearity(df, folder_path)
    print("\nFeatures Numéricas recomendadas (3 Classes):")
    print(safe_features)
    print("Fase 2 de 3 classes finalizada!")
