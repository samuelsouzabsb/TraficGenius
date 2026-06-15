# -*- coding: utf-8 -*-
"""
Script de Consolidação e Avaliação Comparativa de Todos os Modelos
Lê as métricas geradas por cada configuração e cria uma visualização unificada de desempenho.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def apply_chart_style():
    plt.style.use('dark_background')
    plt.rcParams['figure.facecolor'] = '#121214'
    plt.rcParams['axes.facecolor'] = '#1a1a1e'
    plt.rcParams['grid.color'] = '#2d2d34'
    plt.rcParams['font.sans-serif'] = 'sans-serif'
    plt.rcParams['font.family'] = 'sans-serif'

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(project_root, "dataset")
    
    files = {
        'Baseline': 'comparativo_modelos_binaria.csv',
        'Temporal (2020+)': 'comparativo_modelos_desde2020.csv',
        'EUA (Separado)': 'comparativo_modelos_us.csv',
        'Brasil (Separado)': 'comparativo_modelos_br.csv',
        'Flag de País': 'comparativo_modelos_flag_pais.csv'
    }
    
    dfs = {}
    for name, filename in files.items():
        filepath = os.path.join(dataset_dir, filename)
        if os.path.exists(filepath):
            dfs[name] = pd.read_csv(filepath)
            print(f"[INFO] Carregadas métricas de {name}.")
        else:
            print(f"[AVISO] Arquivo {filename} não encontrado.")
            
    if not dfs:
        print("[ERRO] Nenhum arquivo de métricas encontrado. Abortando.")
        return
        
    # Consolidar resultados do Stacking Ensemble
    stacking_results = []
    best_results = []
    
    for config_name, df in dfs.items():
        # Filtra o Stacking Ensemble
        df_stack = df[df['Modelo'] == 'Stacking Ensemble']
        if not df_stack.empty:
            row = df_stack.iloc[0].to_dict()
            row['Configuracao'] = config_name
            stacking_results.append(row)
            
        # Filtra o melhor modelo geral por F1-Score da Classe Grave (1)
        df_best = df.sort_values(by='F1-Score Classe 1', ascending=False).iloc[0]
        row_best = df_best.to_dict()
        row_best['Configuracao'] = config_name
        best_results.append(row_best)
        
    df_stacking = pd.DataFrame(stacking_results)
    df_best = pd.DataFrame(best_results)
    
    # Exibe no console
    print("\n=== COMPARATIVO: STACKING ENSEMBLE POR CONFIGURAÇÃO ===")
    print(df_stacking[['Configuracao', 'Limiar Ótimo', 'Acurácia Global (Ótimo)', 'Acurácia Balanceada', 'F1-Score Classe 1', 'ROC-AUC']].to_string(index=False))
    
    print("\n=== COMPARATIVO: MELHOR MODELO POR CONFIGURAÇÃO ===")
    print(df_best[['Configuracao', 'Modelo', 'Limiar Ótimo', 'Acurácia Global (Ótimo)', 'F1-Score Classe 1', 'ROC-AUC']].to_string(index=False))
    
    # Salva CSVs de consolidação
    df_stacking.to_csv(os.path.join(dataset_dir, "consolidacao_stacking.csv"), index=False)
    df_best.to_csv(os.path.join(dataset_dir, "consolidacao_melhor_modelo.csv"), index=False)
    
    # Plota gráfico comparativo das configurações (usando Stacking Ensemble)
    apply_chart_style()
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    x = range(len(df_stacking))
    tick_labels = df_stacking['Configuracao']
    
    color1 = '#ff007f'
    ax1.set_xlabel('Configuração do Treino', color='#e2e2e9', fontsize=12, labelpad=15)
    ax1.set_ylabel('F1-Score Classe Grave (%)', color=color1, fontsize=12)
    bars = ax1.bar([i - 0.2 for i in x], df_stacking['F1-Score Classe 1'] * 100, width=0.4, label='F1-Score Classe 1 (%)', color=color1, alpha=0.85)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tick_labels, color='#e2e2e9', fontsize=11)
    
    # Adiciona valores sobre as barras de F1
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.8, f"{yval:.2f}%", ha='center', va='bottom', color=color1, fontweight='bold', fontsize=9)
        
    ax2 = ax1.twinx()
    color2 = '#00f0ff'
    ax2.set_ylabel('ROC-AUC (%)', color=color2, fontsize=12)
    bars2 = ax2.bar([i + 0.2 for i in x], df_stacking['ROC-AUC'] * 100, width=0.4, label='ROC-AUC (%)', color=color2, alpha=0.85)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Adiciona valores sobre as barras de AUC
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.8, f"{yval:.2f}%", ha='center', va='bottom', color=color2, fontweight='bold', fontsize=9)
        
    plt.title('Desempenho Geral do Stacking Ensemble por Configuração', fontsize=14, color='#00f0ff', pad=20)
    fig.tight_layout()
    plt.savefig(os.path.join(dataset_dir, "comparativo_geral_configuracoes.png"), dpi=300, facecolor='#121214')
    plt.close()
    
    # Gera o Relatório Markdown consolidado
    relatorio_path = os.path.join(project_root, "relatorio_comparativo_final.md")
    
    markdown_content = f"""# Relatório de Avaliação Comparativa de Novos Treinamentos

Este relatório resume os resultados de desempenho obtidos ao treinar o modelo de classificação de severidade de acidentes (**Leve/Moderado** [0] vs. **Grave/Fatal** [1]) sob diferentes configurações:

1. **Baseline**: Todo o histórico unificado (US + BR), sem flag de país.
2. **Temporal (2020+)**: Apenas acidentes de 2020 em diante (US + BR).
3. **EUA (Separado)**: Modelo treinado exclusivamente com dados dos EUA.
4. **Brasil (Separado)**: Modelo treinado exclusivamente com dados do Brasil.
5. **Flag de País**: Todo o histórico unificado, utilizando uma flag de país (`pais_US`) como variável preditora.

---

## 📊 Desempenho do Stacking Ensemble por Configuração

O Stacking Ensemble combina as previsões dos 5 classificadores base (Regressão Logística, LDA, Random Forest, Rede Neural MLP e XGBoost) através de um meta-aprendedor Logit.

| Configuração | Limiar Ótimo | Acurácia Global (Ótimo) | Acurácia Balanceada | F1-Score Classe Grave (1) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    
    for _, row in df_stacking.iterrows():
        markdown_content += f"| **{row['Configuracao']}** | {row['Limiar Ótimo']:.2f} | {row['Acurácia Global (Ótimo)']*100:.2f}% | {row['Acurácia Balanceada']*100:.2f}% | {row['F1-Score Classe 1']*100:.2f}% | {row['ROC-AUC']*100:.2f}% |\n"
        
    markdown_content += """
---

## 🏆 Melhor Algoritmo Individual por Configuração

Além do Stacking Ensemble, esta tabela mostra o algoritmo individual de melhor desempenho (baseado no F1-Score da Classe Grave) em cada treino.

| Configuração | Algoritmo Vencedor | Limiar Ótimo | Acurácia Global (Ótimo) | F1-Score Classe Grave (1) | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: |
"""

    for _, row in df_best.iterrows():
        markdown_content += f"| **{row['Configuracao']}** | {row['Modelo']} | {row['Limiar Ótimo']:.2f} | {row['Acurácia Global (Ótimo)']*100:.2f}% | {row['F1-Score Classe 1']*100:.2f}% | {row['ROC-AUC']*100:.2f}% |\n"
        
    markdown_content += f"""
---

## 🔍 Principais Conclusões Estatísticas

1. **Segmentação Geográfica (EUA vs. Brasil)**:
   - Os modelos específicos revelam se a dinâmica de acidentes é distinta. O modelo brasileiro costuma sofrer com menor volume de dados, mas pode revelar fatores locais diferentes dos EUA (como o impacto de postos de combustível, velocidade média ou infraestrutura rodoviária).
   
2. **Impacto da Flag de País**:
   - A inclusão da flag `pais_US` no modelo unificado serve como um calibrador de viés geográfico. Se a flag tiver alta importância no XGBoost, ela indica uma diferença estrutural significativa na proporção de acidentes graves entre os países.

3. **Recorte Temporal (Desde 2020)**:
   - Avalia se o comportamento dos acidentes mudou significativamente pós-pandemia ou com as novas configurações de tráfego, servindo como teste de conceito de adaptação temporal.

O gráfico de comparação geral foi gerado em `dataset/comparativo_geral_configuracoes.png`.
"""
    
    with open(relatorio_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
        
    print(f"\n[INFO] Relatório comparativo gravado em {relatorio_path}")

if __name__ == "__main__":
    main()
