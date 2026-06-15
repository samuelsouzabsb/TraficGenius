# -*- coding: utf-8 -*-
"""
Script de Análise: Influência da Clusterização Espacial no Modelo de Severidade
Este script investiga o impacto do 'Cluster_Espacial' na performance do modelo preditivo
e analisa a distribuição de características de acidentes entre as diferentes zonas.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

# Configuração de estilo visual dark/neon do TraficGenius
def apply_chart_style():
    plt.style.use('dark_background')
    plt.rcParams['figure.facecolor'] = '#121214'
    plt.rcParams['axes.facecolor'] = '#1a1a1e'
    plt.rcParams['grid.color'] = '#2d2d34'
    plt.rcParams['font.sans-serif'] = 'sans-serif'
    plt.rcParams['font.family'] = 'sans-serif'

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(project_root, "dataset", "dataset_amostra_limpa_avancado.parquet")
    
    print("Carregando o dataset limpo...")
    df = pd.read_parquet(dataset_path)
    print(f"Dataset carregado com {len(df):,} linhas.")
    
    # Amostra de 200.000 linhas se o dataset for muito grande, para garantir velocidade de execução
    if len(df) > 200000:
        print("Fazendo amostragem de 200.000 registros para otimizar tempo de análise...")
        df_analysis = df.sample(n=200000, random_state=42).copy()
    else:
        df_analysis = df.copy()

    # Mapear variáveis categóricas se houver
    drop_cols = ['Sensacao_Termica_F', 'Nascer_Por_Sol']
    for col in drop_cols:
        if col in df_analysis.columns:
            df_analysis = df_analysis.drop(columns=[col])

    # 1. Caracterização dos Clusters
    print("\n=== Analisando as características por Cluster Espacial ===")
    
    # Proporções de severidade por cluster
    sev_by_cluster = pd.crosstab(df_analysis['Cluster_Espacial'], df_analysis['Severidade'], normalize='index') * 100
    print("\nDistribuição de Severidade (%) por Cluster Espacial:")
    print(sev_by_cluster.round(2))
    
    # Fatores de infraestrutura por cluster (médias)
    infra_cols = ['Semaforo', 'Cruzamento', 'Juncao', 'Pare', 'Lombada', 'Redutor_Velocidade']
    available_infra = [c for c in infra_cols if c in df_analysis.columns]
    infra_by_cluster = df_analysis.groupby('Cluster_Espacial')[available_infra].mean() * 100
    print("\nPresença de Infraestrutura (%) por Cluster Espacial:")
    print(infra_by_cluster.round(2))
    
    # Clima médio por cluster
    weather_cols = ['Temperatura_F', 'Visibilidade_Milhas', 'Precipitacao_Polegadas', 'Umidade_Percentual']
    available_weather = [c for c in weather_cols if c in df_analysis.columns]
    weather_by_cluster = df_analysis.groupby('Cluster_Espacial')[available_weather].mean()
    print("\nMédias de Variáveis Meteorológicas por Cluster Espacial:")
    print(weather_by_cluster.round(2))
    
    # 2. Geração de Gráficos
    print("\nGerando gráficos de análise...")
    apply_chart_style()
    
    # Gráfico 1: Severidade por Cluster (Empilhado)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#00ff66', '#00f0ff', '#ff007f', '#ff3300'] # Verde, Ciano, Rosa, Vermelho
    sev_by_cluster.plot(kind='bar', stacked=True, color=colors, ax=ax)
    ax.set_title('Distribuição de Severidade dos Acidentes por Cluster Espacial', fontsize=14, color='#00f0ff', pad=15)
    ax.set_xlabel('ID do Cluster Espacial', fontsize=12, color='#e2e2e9')
    ax.set_ylabel('Percentual (%)', fontsize=12, color='#e2e2e9')
    ax.legend(title='Severidade', facecolor='#1a1a1e', edgecolor='#2d2d34', labelcolor='#e2e2e9')
    plt.tight_layout()
    plot_sev_path = os.path.join(project_root, "distribuicao_severidade_por_cluster.png")
    plt.savefig(plot_sev_path, dpi=300, bbox_inches='tight', facecolor='#121214')
    plt.close()
    
    # Gráfico 2: Calor de Infraestrutura por Cluster
    plt.figure(figsize=(12, 6))
    sns.heatmap(infra_by_cluster, annot=True, fmt=".1f", cmap=sns.dark_palette('#00f0ff', as_cmap=True), 
                cbar_kws={'label': 'Prevalência (%)'}, linewidths=0.5, linecolor='#121214')
    plt.title('Prevalência de Elementos de Infraestrutura por Cluster Espacial', fontsize=14, color='#00f0ff', pad=15)
    plt.xlabel('Variáveis de Infraestrutura', fontsize=12, color='#e2e2e9')
    plt.ylabel('ID do Cluster Espacial', fontsize=12, color='#e2e2e9')
    plt.tight_layout()
    plot_infra_path = os.path.join(project_root, "caracterizacao_infra_por_cluster.png")
    plt.savefig(plot_infra_path, dpi=300, bbox_inches='tight', facecolor='#121214')
    plt.close()

    # 3. Experimento de Modelagem (Impacto do Cluster Espacial)
    print("\n=== Iniciando o Experimento de Modelagem (com vs sem recursos espaciais) ===")
    
    # Definindo conjuntos de features para teste
    all_features = [c for c in df_analysis.columns if c != 'Severidade']
    
    features_setups = {
        "1. Completo (Coords + Cluster)": all_features,
        "2. Apenas Coordenadas (Sem Cluster)": [c for c in all_features if c != 'Cluster_Espacial'],
        "3. Apenas Cluster (Sem Coords)": [c for c in all_features if c not in ['Latitude_Inicial', 'Longitude_Inicial']],
        "4. Sem Dados Espaciais (Sem Coords/Cluster)": [c for c in all_features if c not in ['Latitude_Inicial', 'Longitude_Inicial', 'Cluster_Espacial']]
    }
    
    results = {}
    
    y = df_analysis['Severidade'] - 1
    
    for setup_name, feat_list in features_setups.items():
        print(f"\nTreinando modelo para setup: {setup_name}...")
        print(f"Número de features: {len(feat_list)}")
        
        X = df_analysis[feat_list]
        
        # Split estratificado
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        # Escalar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Amostras balanceadas por pesos
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        
        # Modelo XGBoost rápido
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            eval_metric='mlogloss',
            tree_method='hist',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        
        # Avaliar
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred) * 100
        f1_macro = f1_score(y_test, y_pred, average='macro') * 100
        
        # Importância da feature 'Cluster_Espacial' se existir
        cluster_importance = 0.0
        if 'Cluster_Espacial' in feat_list:
            feat_importances = model.feature_importances_
            cluster_idx = feat_list.index('Cluster_Espacial')
            cluster_importance = feat_importances[cluster_idx] * 100
            
        print(f"-> Acurácia: {acc:.2f}% | F1-Score Macro: {f1_macro:.2f}%")
        if 'Cluster_Espacial' in feat_list:
            print(f"-> Importância Relativa do 'Cluster_Espacial': {cluster_importance:.2f}%")
            
        results[setup_name] = {
            "Accuracy": acc,
            "F1_Macro": f1_macro,
            "Cluster_Importance": cluster_importance
        }
        
    # Gráfico 3: Comparação de Performance
    apply_chart_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    setups = list(results.keys())
    f1_vals = [results[s]['F1_Macro'] for s in setups]
    acc_vals = [results[s]['Accuracy'] for s in setups]
    
    x = np.arange(len(setups))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, f1_vals, width, label='F1-Score Macro (%)', color='#ff007f')
    rects2 = ax.bar(x + width/2, acc_vals, width, label='Acurácia Global (%)', color='#00f0ff')
    
    ax.set_title('Influência dos Dados Espaciais e Clusterização no Desempenho do Modelo', fontsize=14, color='#00f0ff', pad=15)
    ax.set_ylabel('Pontuação (%)', fontsize=12, color='#e2e2e9')
    ax.set_xticks(x)
    ax.set_xticklabels(setups, rotation=15, color='#e2e2e9')
    ax.tick_params(colors='#e2e2e9')
    ax.set_ylim(min(f1_vals) - 5, max(acc_vals) + 5)
    ax.legend(facecolor='#1a1a1e', edgecolor='#2d2d34', labelcolor='#e2e2e9')
    
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, color='#e2e2e9')
                    
    plt.tight_layout()
    plot_compare_path = os.path.join(project_root, "comparacao_desempenho_espacial.png")
    plt.savefig(plot_compare_path, dpi=300, bbox_inches='tight', facecolor='#121214')
    plt.close()
    
    print("\nAnálise completa finalizada com sucesso! Arquivos de imagem salvos no projeto.")
