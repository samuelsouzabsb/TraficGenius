# -*- coding: utf-8 -*-
"""
Pipeline Fase 2: Testes de Premissas Estatísticas e Multicolinearidade (Binário)
Atua como Cientista de Dados Sênior realizando validação estatística multivariada.
Todos os nomes de colunas estão traduzidos para o português.
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, kstest, levene, norm
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

def apply_chart_style():
    """
    Identidade visual do TraficGenius (neon escura).
    """
    plt.style.use('dark_background')
    plt.rcParams['figure.facecolor'] = '#121214'
    plt.rcParams['axes.facecolor'] = '#1a1a1e'
    plt.rcParams['grid.color'] = '#2d2d34'
    plt.rcParams['font.sans-serif'] = 'sans-serif'
    plt.rcParams['font.family'] = 'sans-serif'

def test_normality_homoscedasticity(df):
    """
    Executa testes de Normalidade (Shapiro-Wilk e Kolmogorov-Smirnov)
    e de Homocedasticidade (Levene) entre os dois grupos de severidade.
    """
    print("--- 1. Testes de Normalidade e Homocedasticidade (Binário) ---")
    
    num_cols = [
        'temperatura_celsius', 'ponto_orvalho_celsius', 'pressao_hpa', 
        'velocidade_vento_u', 'velocidade_vento_v',
        'cobertura_nuvens_percentual', 'precipitacao_milimetros', 
        'velocidade_media', 'quantidade_faixas_media', 
        'curvatura_acumulada', 'desvio_maximo_curvatura', 
        'extensao_rodovia_metros_res11', 'extensao_rodovia_metros_res10', 'extensao_rodovia_metros_res9',
        'interacao_velocidade_faixas', 'interacao_chuva_curvas', 'interacao_clima_curvatura'
    ]
    
    # Amostras para os testes estatísticos (Shapiro é limitado a N <= 5000)
    df_shapiro = df.sample(n=4500, random_state=42)
    df_ks = df.sample(n=10000, random_state=42)
    
    print("\n[Normalidade] Testes de Aderência à Normalidade (p > 0.05 sugere normalidade):")
    for col in num_cols:
        # Shapiro-Wilk
        stat_s, p_s = shapiro(df_shapiro[col])
        # Kolmogorov-Smirnov contra distribuição normal teórica com mesma média/desvio
        mean, std = df_ks[col].mean(), df_ks[col].std()
        if std == 0:
            std = 1e-6
        stat_k, p_k = kstest(df_ks[col], 'norm', args=(mean, std))
        
        is_normal_s = "Sim" if p_s > 0.05 else "Não"
        is_normal_k = "Sim" if p_k > 0.05 else "Não"
        print(f" - {col:30}: Shapiro P={p_s:.6f} ({is_normal_s}) | K-S P={p_k:.6f} ({is_normal_k})")
        
    print("\n[Homocedasticidade] Teste de Levene para Igualdade de Variâncias (Leve/Moderado vs Grave/Fatal):")
    group_0 = df[df['severidade_binaria'] == 0]
    group_1 = df[df['severidade_binaria'] == 1]
    
    for col in num_cols:
        stat, p = levene(group_0[col], group_1[col])
        is_homosc = "Sim" if p > 0.05 else "Não"
        print(f" - {col:30}: Levene P-Valor={p:.6f} | Homocedástico? {is_homosc}")

def plot_normality_distributions(df, output_dir):
    """
    Plota KDEs reais vs Gaussiana teórica para as principais variáveis contínuas.
    """
    apply_chart_style()
    print("\nGerando gráfico comparativo de normalidade (KDE vs Gaussiana)...")
    
    plot_cols = ['temperatura_celsius', 'pressao_hpa', 'velocidade_media', 'quantidade_faixas_media']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    df_sample = df.sample(n=10000, random_state=42)
    
    for idx, col in enumerate(plot_cols):
        ax = axes[idx]
        data = df_sample[col].dropna()
        
        sns.kdeplot(data, ax=ax, fill=True, color='#ff007f', alpha=0.3, label='Distribuição Real', linewidth=2)
        mean, std = data.mean(), data.std()
        if std == 0:
            std = 1e-6
        x_axis = np.linspace(data.min(), data.max(), 100)
        ax.plot(x_axis, norm.pdf(x_axis, mean, std), color='#00f0ff', linestyle='--', linewidth=2, label='Normal Teórica')
        
        ax.set_title(f"Aderência à Normalidade: {col}", color='#00f0ff', fontsize=12, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("Densidade", color='#e2e2e9')
        ax.tick_params(colors='#e2e2e9')
        ax.legend(facecolor='#1a1a1e', edgecolor='#2d2d34', fontsize=9, labelcolor='#e2e2e9')
        
    plt.tight_layout()
    out_path = os.path.join(output_dir, "distribuicao_normalidade_binaria.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#121214')
    plt.close()
    print(f"Gráfico salvo em: {out_path}")

def check_and_plot_correlation(df, output_dir):
    """
    Plota a matriz de correlação linear (Pearson) das variáveis numéricas.
    """
    apply_chart_style()
    print("\n--- 2. Matriz de Correlação (Pearson) ---")
    
    num_cols = [
        'temperatura_celsius', 'ponto_orvalho_celsius', 'pressao_hpa', 
        'velocidade_vento_u', 'velocidade_vento_v',
        'cobertura_nuvens_percentual', 'precipitacao_milimetros', 
        'velocidade_media', 'quantidade_faixas_media', 
        'curvatura_acumulada', 'desvio_maximo_curvatura', 'quantidade_curvas_acentuadas', 
        'quantidade_cruzamentos', 'quantidade_semaforos', 'quantidade_rotatorias', 
        'quantidade_pontes', 'quantidade_tuneis', 'quantidade_postos_combustivel', 
        'quantidade_restaurantes', 'quantidade_escolas', 'quantidade_hospitais', 
        'quantidade_rodovias_distintas', 'total_curvas_acentuadas', 'quantidade_locais_interesse', 
        'area_urbana_m2', 'area_rural_m2', 'Hora_do_Dia', 'Dia_da_Semana', 'Mes',
        'interacao_velocidade_faixas', 'interacao_chuva_curvas', 'interacao_clima_curvatura'
    ]
    
    existing_cols = [c for c in num_cols if c in df.columns]
    corr_matrix = df[existing_cols].corr(method='pearson')
    
    plt.figure(figsize=(16, 14))
    cmap = sns.diverging_palette(180, 340, as_cmap=True)
    sns.heatmap(corr_matrix, annot=False, cmap=cmap, vmin=-1, vmax=1,
                cbar_kws={'shrink': 0.8}, linewidths=0.2, linecolor='#121214')
    
    plt.title("Matriz de Correlação (Pearson) - Atributos Contínuos e de Contagem", fontsize=16, color='#00f0ff', pad=20)
    plt.xticks(rotation=45, ha='right', color='#e2e2e9', fontsize=9)
    plt.yticks(color='#e2e2e9', fontsize=9)
    
    out_path = os.path.join(output_dir, "matriz_correlacao_binaria.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#121214')
    plt.close()
    print(f"Gráfico de matriz de correlação salvo em: {out_path}")

def check_multicollinearity(df, output_dir):
    """
    Calcula iterativamente o VIF das variáveis contínuas numéricas e
    remove aquelas com VIF > 10.
    """
    print("\n--- 3. Análise Iterativa de Multicolinearidade (VIF) ---")
    
    num_cols = [
        'temperatura_celsius', 'ponto_orvalho_celsius', 'pressao_hpa', 
        'velocidade_vento_u', 'velocidade_vento_v',
        'cobertura_nuvens_percentual', 'precipitacao_milimetros', 
        'velocidade_media', 'quantidade_faixas_media', 
        'curvatura_acumulada', 'desvio_maximo_curvatura', 'quantidade_curvas_acentuadas',
        'quantidade_cruzamentos', 'quantidade_semaforos', 'quantidade_rotatorias', 
        'quantidade_pontes', 'quantidade_tuneis', 'quantidade_postos_combustivel', 
        'quantidade_restaurantes', 'quantidade_escolas', 'quantidade_hospitais', 
        'quantidade_rodovias_distintas', 'total_curvas_acentuadas', 'quantidade_locais_interesse', 
        'area_urbana_m2', 'area_rural_m2', 'extensao_rodovia_metros_res11', 
        'extensao_rodovia_metros_res10', 'extensao_rodovia_metros_res9',
        'Hora_do_Dia', 'Dia_da_Semana', 'Mes',
        'interacao_velocidade_faixas', 'interacao_chuva_curvas', 'interacao_clima_curvatura'
    ]
    
    # Amostra para o cálculo do VIF devido ao custo O(N*P^2)
    X = df[num_cols].dropna().sample(n=100000, random_state=42)
    
    # Adiciona pequena constante diagonal às variâncias para estabilidade numérica
    X_scaled = (X - X.mean()) / (X.std().replace(0, 1))
    
    vif_initial = []
    for col in X_scaled.columns:
        y_vif = X_scaled[col]
        X_vif = X_scaled.drop(columns=[col])
        model = LinearRegression()
        model.fit(X_vif, y_vif)
        r_sq = model.score(X_vif, y_vif)
        vif = 1 / (1 - r_sq) if r_sq < 1 else float('inf')
        vif_initial.append((col, vif))
        
    vif_initial_df = pd.DataFrame(vif_initial, columns=['feature', 'VIF']).sort_values(by="VIF", ascending=True)
    
    # Plota VIF inicial
    plt.figure(figsize=(12, 10))
    colors = ['#ff0055' if v > 10 else '#00f0ff' for v in vif_initial_df['VIF']]
    bars = plt.barh(vif_initial_df['feature'], vif_initial_df['VIF'], color=colors, height=0.6)
    plt.axvline(x=10, color='#ffcc00', linestyle='--', linewidth=1.5, label='Limite Crítico (VIF = 10)')
    
    plt.title("VIF - Análise de Multicolinearidade Inicial (Binária)", fontsize=14, color='#00f0ff', pad=15)
    plt.xlabel("Valor do VIF", fontsize=12, color='#e2e2e9')
    plt.xticks(color='#e2e2e9')
    plt.yticks(color='#e2e2e9', fontsize=8)
    plt.legend(facecolor='#1a1a1e', edgecolor='#2d2d34', labelcolor='#e2e2e9')
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.2, bar.get_y() + bar.get_height()/2, 
                 f'{width:.2f}', 
                 va='center', ha='left', fontsize=8, color='#e2e2e9')
                 
    out_path = os.path.join(output_dir, "vif_multicolinearidade_binaria.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#121214')
    plt.close()
    print(f"Gráfico de VIF inicial salvo em: {out_path}")
    
    # Redução stepwise iterativa
    removed_features = []
    X_stepwise = X_scaled.copy()
    
    while True:
        vif_list = []
        for col in X_stepwise.columns:
            y_vif = X_stepwise[col]
            X_vif = X_stepwise.drop(columns=[col])
            model = LinearRegression()
            model.fit(X_vif, y_vif)
            r_sq = model.score(X_vif, y_vif)
            vif = 1 / (1 - r_sq) if r_sq < 1 else float('inf')
            vif_list.append((col, vif))
            
        vif_df = pd.DataFrame(vif_list, columns=['feature', 'VIF']).sort_values(by="VIF", ascending=False)
        max_vif = vif_df.iloc[0]['VIF']
        max_feature = vif_df.iloc[0]['feature']
        
        if max_vif > 10.0:
            print(f" - Removendo '{max_feature}' por VIF crítico: {max_vif:.2f}")
            X_stepwise = X_stepwise.drop(columns=[max_feature])
            removed_features.append(max_feature)
        else:
            break
            
    selected_numeric_features = list(X_stepwise.columns)
    print(f"Total de atributos contínuos selecionados após VIF stepwise: {len(selected_numeric_features)}")
    print(f"Atributos removidos: {removed_features}")
    
    return selected_numeric_features

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(project_root, "dataset", "dataset_amostra_limpa_binaria.parquet")
    output_dir = os.path.join(project_root, "dataset")
    
    print("Carregando dataset limpo...")
    df = pd.read_parquet(dataset_path)
    
    # 1. Testes de Normalidade e Homocedasticidade
    test_normality_homoscedasticity(df)
    
    # 2. Distribuições de Normalidade
    plot_normality_distributions(df, output_dir)
    
    # 3. Matriz de Correlação
    check_and_plot_correlation(df, output_dir)
    
    # 4. Redução stepwise por VIF
    selected_numeric = check_multicollinearity(df, output_dir)
    
    # Categorias traduzidas
    categorical_features = ['rodovia_dominante_res11', 'rodovia_dominante_res10', 'rodovia_dominante_res9']
    temporal_features = ['Horario_Pico']
    
    # Lista final
    final_features = selected_numeric + categorical_features + temporal_features
    
    output_json = os.path.join(output_dir, "features_selecionadas_binaria.json")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(final_features, f, ensure_ascii=False, indent=4)
        
    print(f"\nLista de features selecionadas persistida em: {output_json}")
    print(f"Lista de variáveis de entrada para o modelo ({len(final_features)}): {final_features}")
