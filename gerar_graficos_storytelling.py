# -*- coding: utf-8 -*-
"""
Script de Geração de Gráficos para Storytelling Estatístico
Gera visualizações premium de EDA, premissas clássicas, ROC/PR, calibração, t-SNE, SHAP, Lift e Ganho.
Salva todos os gráficos na pasta de artefatos.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.manifold import TSNE
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf

# Tenta carregar o TensorFlow de forma segura
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except (ImportError, Exception):
    TENSORFLOW_AVAILABLE = False

# Tenta carregar o SHAP de forma segura
try:
    import shap
    SHAP_AVAILABLE = True
except (ImportError, Exception):
    SHAP_AVAILABLE = False

warnings.filterwarnings('ignore')

def apply_chart_style():
    plt.style.use('dark_background')
    plt.rcParams['figure.facecolor'] = '#121214'
    plt.rcParams['axes.facecolor'] = '#1a1a1e'
    plt.rcParams['grid.color'] = '#2d2d34'
    plt.rcParams['font.sans-serif'] = 'sans-serif'
    plt.rcParams['font.family'] = 'sans-serif'

def get_continuous_features(df):
    cols = [
        'quantidade_cruzamentos', 'quantidade_semaforos', 'velocidade_media',
        'quantidade_faixas_media', 'curvatura_acumulada', 'desvio_maximo_curvatura',
        'quantidade_curvas_acentuadas', 'quantidade_rotatorias', 'quantidade_pontes',
        'comprimento_pontes_metros', 'quantidade_tuneis', 'comprimento_tuneis_metros',
        'quantidade_postos_combustivel', 'quantidade_restaurantes', 'quantidade_escolas',
        'quantidade_hospitais', 'quantidade_rodovias_distintas', 'total_curvas_acentuadas',
        'quantidade_locais_interesse', 'area_urbana_m2', 'area_rural_m2',
        'temperatura_celsius', 'ponto_orvalho_celsius', 'pressao_hpa',
        'velocidade_vento_u', 'velocidade_vento_v', 'cobertura_nuvens_percentual',
        'precipitacao_milimetros'
    ]
    return [c for c in cols if c in df.columns]

def plot_lift_gain_curves(y_true, y_prob_dict, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Prepara dados de baseline
    n_samples = len(y_true)
    n_positives = np.sum(y_true)
    baseline_rate = n_positives / n_samples
    
    # Eixo X comum (percentual da população)
    percentiles = np.linspace(0, 100, 101)
    
    for name, y_prob in y_prob_dict.items():
        # Ordena descendente por probabilidade
        sorted_indices = np.argsort(y_prob)[::-1]
        y_true_sorted = y_true.iloc[sorted_indices].to_numpy()
        
        # Cumulative Gain
        cum_gains = np.cumsum(y_true_sorted) / n_positives * 100
        cum_gains = np.insert(cum_gains, 0, 0) # Adiciona ponto zero
        x_gains = np.linspace(0, 100, len(cum_gains))
        
        # Interpola para percentis fixos
        gains_interp = np.interp(percentiles, x_gains, cum_gains)
        ax1.plot(percentiles, gains_interp, label=name, lw=2)
        
        # Lift Curve
        lift_interp = np.zeros_like(percentiles)
        # Ponto zero tem lift infinito ou indefinido, colocamos o mesmo do primeiro percentil
        lift_interp[1:] = (gains_interp[1:] / 100) / (percentiles[1:] / 100)
        lift_interp[0] = lift_interp[1]
        ax2.plot(percentiles, lift_interp, label=name, lw=2)
        
    # Gráfico de Ganho Acumulado
    ax1.plot([0, 100], [0, 100], 'k--', color='#e2e2e9', label='Modelo Aleatório')
    ax1.set_title('Curva de Ganho Acumulado (Cumulative Gain)', color='#00f0ff', fontsize=12, pad=15)
    ax1.set_xlabel('% da População Ordenada por Probabilidade', color='#e2e2e9')
    ax1.set_ylabel('% de Casos Graves Detectados', color='#e2e2e9')
    ax1.legend(facecolor='#1a1a1e', labelcolor='#e2e2e9')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Gráfico de Lift
    ax2.axhline(y=1.0, color='r', linestyle='--', label='Linha de Referência (Lift = 1)')
    ax2.set_title('Curva de Lift', color='#ff007f', fontsize=12, pad=15)
    ax2.set_xlabel('% da População Ordenada por Probabilidade', color='#e2e2e9')
    ax2.set_ylabel('Lift (Ganho sobre o Acaso)', color='#e2e2e9')
    ax2.legend(facecolor='#1a1a1e', labelcolor='#e2e2e9')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor='#121214')
    plt.close()

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    # Os arquivos salvos na pasta de artefatos da conversa do Gemini
    artifacts_dir = "C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7"
    
    input_file = os.path.join(project_root, "dataset", "dataset_amostra_limpa_binaria.parquet")
    output_dir = os.path.join(project_root, "dataset")
    
    apply_chart_style()
    print("[INFO] Carregando dados...")
    df = pd.read_parquet(input_file)
    df['severidade_binaria'] = df['severidade_binaria'].astype(int)
    continuous_cols = get_continuous_features(df)
    
    # -------------------------------------------------------------
    # 1. Gráficos de Análise Exploratória (EDA)
    # -------------------------------------------------------------
    print("\n--- Gerando Gráficos EDA ---")
    
    # KDE Distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.kdeplot(data=df, x='velocidade_media', hue='severidade_binaria', fill=True, alpha=0.4, 
                palette=['#00f0ff', '#ff007f'], common_norm=False, ax=ax1)
    ax1.set_title('Distribuição da Velocidade Média por Severidade', color='#00f0ff', fontsize=12, pad=15)
    ax1.set_xlabel('Velocidade Média (km/h)', color='#e2e2e9')
    ax1.set_ylabel('Densidade', color='#e2e2e9')
    
    sns.kdeplot(data=df, x='temperatura_celsius', hue='severidade_binaria', fill=True, alpha=0.4, 
                palette=['#00f0ff', '#ff007f'], common_norm=False, ax=ax2)
    ax2.set_title('Distribuição da Temperatura por Severidade', color='#ff007f', fontsize=12, pad=15)
    ax2.set_xlabel('Temperatura (Celsius)', color='#e2e2e9')
    ax2.set_ylabel('Densidade', color='#e2e2e9')
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, "story_eda_kde.png"), dpi=300, facecolor='#121214')
    plt.close()
    
    # Boxplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=df, x='severidade_binaria', y='velocidade_media', palette=['#00f0ff', '#ff007f'], ax=ax1)
    ax1.set_title('Dispersão de Velocidade Média por Severidade', color='#00f0ff', fontsize=12)
    ax1.set_xticklabels(['Leve/Moderado', 'Grave/Fatal'], color='#e2e2e9')
    ax1.set_xlabel('Severidade', color='#e2e2e9')
    ax1.set_ylabel('Velocidade Média (km/h)', color='#e2e2e9')
    
    sns.boxplot(data=df, x='severidade_binaria', y='temperatura_celsius', palette=['#00f0ff', '#ff007f'], ax=ax2)
    ax2.set_title('Dispersão de Temperatura por Severidade', color='#ff007f', fontsize=12)
    ax2.set_xticklabels(['Leve/Moderado', 'Grave/Fatal'], color='#e2e2e9')
    ax2.set_xlabel('Severidade', color='#e2e2e9')
    ax2.set_ylabel('Temperatura (Celsius)', color='#e2e2e9')
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, "story_eda_boxplots.png"), dpi=300, facecolor='#121214')
    plt.close()
    
    # Correlation Heatmap (Top 10)
    plt.figure(figsize=(10, 8))
    corr_mat = df[continuous_cols].corr()
    # Pega top 10 features correlacionadas com a severidade
    corr_with_target = df[continuous_cols + ['severidade_binaria']].corr()['severidade_binaria'].abs().sort_values(ascending=False)
    top_cols = corr_with_target.index[1:11].tolist()
    
    sns.heatmap(df[top_cols].corr(), annot=True, fmt='.2f', cmap='magma', 
                cbar_kws={'label': 'Coeficiente de Correlação'}, linewidths=0.5, linecolor='#121214')
    plt.title('Matriz de Correlação (Top 10 Variáveis Correlacionadas)', color='#00f0ff', fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, "story_eda_correlation.png"), dpi=300, facecolor='#121214')
    plt.close()
    
    # Parallel Coordinates (subsample)
    plt.figure(figsize=(12, 5))
    df_par = df.sample(150, random_state=42)
    cols_par = ['velocidade_media', 'temperatura_celsius', 'quantidade_cruzamentos', 'quantidade_faixas_media']
    df_par_scaled = df_par.copy()
    # Padroniza apenas para visualização paralela
    for c in cols_par:
        df_par_scaled[c] = (df_par[c] - df_par[c].mean()) / df_par[c].std()
    pd.plotting.parallel_coordinates(df_par_scaled[cols_par + ['severidade_binaria']], 'severidade_binaria', 
                                    color=['#00f0ff', '#ff007f'], alpha=0.5, lw=1.5)
    plt.title('Perfis Multivariados de Acidentes (Coordenadas Paralelas)', color='#00f0ff', fontsize=13, pad=15)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, "story_eda_parallel.png"), dpi=300, facecolor='#121214')
    plt.close()
    
    # -------------------------------------------------------------
    # 2. Gráficos de Diagnósticos de Premissas Clássicas
    # -------------------------------------------------------------
    print("\n--- Gerando Gráficos de Diagnósticos ---")
    
    # Amostra para resíduos do modelo OLS
    df_sample = df.sample(5000, random_state=42)
    df_sample['pais_US'] = (df_sample['pais'] == 'US').astype(int)
    cat_cols = ['rodovia_dominante_res11', 'rodovia_dominante_res10', 'rodovia_dominante_res9']
    df_sample_encoded = pd.get_dummies(df_sample, columns=cat_cols, drop_first=True)
    
    target_col = 'severidade_binaria'
    feature_cols = [c for c in df_sample_encoded.columns if c not in [target_col, 'data_inversa', 'pais']]
    
    # Garante inteiros nas dummies
    for c in feature_cols:
        if df_sample_encoded[c].dtype == bool:
            df_sample_encoded[c] = df_sample_encoded[c].astype(int)
            
    X_reg = sm.add_constant(df_sample_encoded[feature_cols].astype(float))
    y_reg = df_sample_encoded[target_col].astype(float)
    ols_model = sm.OLS(y_reg, X_reg).fit()
    residuals = ols_model.resid
    fitted = ols_model.fittedvalues
    
    # Q-Q Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    sm.qqplot(residuals, line='45', fit=True, ax=ax)
    ax.get_lines()[0].set_color('#ff007f')
    ax.get_lines()[0].set_markersize(4)
    ax.get_lines()[1].set_color('#00f0ff')
    ax.get_lines()[1].set_linewidth(2)
    plt.title('Gráfico de Probabilidade Normal (NPP Q-Q Plot) dos Resíduos', color='#00f0ff', fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, "story_premissas_qqplot.png"), dpi=300, facecolor='#121214')
    plt.close()
    
    # Residuals vs Fitted
    plt.figure(figsize=(8, 5))
    plt.scatter(fitted, residuals, alpha=0.3, color='#00f0ff', s=10)
    plt.axhline(y=0, color='#ff007f', linestyle='--', lw=2)
    plt.title('Homocedasticidade: Resíduos vs. Valores Ajustados', color='#00f0ff', fontsize=12, pad=15)
    plt.xlabel('Valores Ajustados (Previsão Linear)', color='#e2e2e9')
    plt.ylabel('Resíduos do Modelo', color='#e2e2e9')
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, "story_premissas_residuals_fitted.png"), dpi=300, facecolor='#121214')
    plt.close()
    
    # ACF Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_acf(residuals, lags=20, ax=ax, color='#00f0ff', vlines_kwargs={"colors": '#ff007f'})
    plt.title('Autocorrelação Espacial/Sequencial dos Erros (Correlograma ACF)', color='#00f0ff', fontsize=12, pad=15)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, "story_premissas_acf.png"), dpi=300, facecolor='#121214')
    plt.close()
    
    # VIF Bar Chart
    diag_json_path = os.path.join(output_dir, "diagnosticos_estatisticos.json")
    if os.path.exists(diag_json_path):
        with open(diag_json_path, 'r') as f:
            diag_data = json.load(f)
        vif_df = pd.DataFrame(diag_data['multicollinearity']['vif_values'])
        vif_df = vif_df[vif_df['feature'] != 'const'].sort_values(by='vif', ascending=True)
        
        plt.figure(figsize=(10, 6))
        colors = ['#ff007f' if v > 10 else '#00f0ff' for v in vif_df['vif']]
        plt.barh(vif_df['feature'], vif_df['vif'], color=colors, height=0.6)
        plt.axvline(x=10.0, color='r', linestyle='--', label='Limite Tolerável (VIF = 10)')
        plt.title('Fator de Inflação da Variância (VIF) dos Preditores', color='#00f0ff', fontsize=13, pad=15)
        plt.xlabel('VIF', color='#e2e2e9')
        plt.legend(facecolor='#1a1a1e', labelcolor='#e2e2e9')
        plt.tight_layout()
        plt.savefig(os.path.join(artifacts_dir, "story_premissas_vif.png"), dpi=300, facecolor='#121214')
        plt.close()
        
    # -------------------------------------------------------------
    # 3. Modelos Preditivos: ROC, PR, Calibração, Lift/Gain, t-SNE e SHAP
    # -------------------------------------------------------------
    print("\n--- Processando Modelos Preditivos para Storytelling ---")
    # Prepara base completa de validação/teste utilizando Flag de País
    df['pais_US'] = (df['pais'] == 'US').astype(int)
    df_eval = df.drop(columns=['data_inversa', 'pais'])
    df_encoded_eval = pd.get_dummies(df_eval, columns=cat_cols, drop_first=True)
    
    # Libera df bruta
    del df
    
    feature_cols_eval = [c for c in df_encoded_eval.columns if c != target_col]
    X_full = df_encoded_eval[feature_cols_eval]
    y_full = df_encoded_eval[target_col]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.3, stratify=y_full, random_state=42
    )
    
    scaler = joblib.load(os.path.join(output_dir, "scaler_flag_pais.joblib"))
    X_test_scaled = scaler.transform(X_test)
    
    # Amostra do teste para predição
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), size=min(15000, len(X_test)), replace=False)
    X_sample_df = X_test.iloc[sample_indices]
    X_sample_scaled = X_test_scaled[sample_indices]
    y_sample = y_test.iloc[sample_indices]
    
    # Carrega modelos
    models = {}
    print("Carregando modelos salvos...")
    models['Logistic Regression'] = joblib.load(os.path.join(output_dir, "model_logit_flag_pais.joblib"))
    models['LDA'] = joblib.load(os.path.join(output_dir, "model_lda_flag_pais.joblib"))
    models['Random Forest'] = joblib.load(os.path.join(output_dir, "model_rf_flag_pais.joblib"))
    models['XGBoost'] = joblib.load(os.path.join(output_dir, "model_xgb_flag_pais.joblib"))
    models['Stacking Ensemble'] = joblib.load(os.path.join(output_dir, "model_stacking_flag_pais.joblib"))
    
    # Carrega MLP
    if TENSORFLOW_AVAILABLE and os.path.exists(os.path.join(output_dir, "model_mlp_flag_pais.h5")):
        models['Neural Network (MLP)'] = tf.keras.models.load_model(os.path.join(output_dir, "model_mlp_flag_pais.h5"))
    elif os.path.exists(os.path.join(output_dir, "model_mlp_flag_pais.joblib")):
        models['Neural Network (MLP)'] = joblib.load(os.path.join(output_dir, "model_mlp_flag_pais.joblib"))
        
    # Gera probabilidades preditas
    probs = {}
    meta_preds_sample = {}
    
    for name, model in models.items():
        if name == 'Stacking Ensemble':
            continue
        print(f"Predizendo probabilities para {name}...")
        if name == 'Neural Network (MLP)' and TENSORFLOW_AVAILABLE:
            p = model.predict(X_sample_scaled).ravel()
        else:
            p = model.predict_proba(X_sample_scaled)[:, 1]
            
        probs[name] = p
        meta_preds_sample[name] = p
            
    # Stacking
    X_meta_sample = pd.DataFrame(meta_preds_sample)
    expected_cols = ['Logistic Regression', 'LDA', 'Random Forest', 'Neural Network (MLP)', 'XGBoost']
    X_meta_sample = X_meta_sample[expected_cols]
    probs['Stacking Ensemble'] = models['Stacking Ensemble'].predict_proba(X_meta_sample)[:, 1]
    
    # A. Curvas ROC
    plt.figure(figsize=(8, 7))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_sample, p)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#e2e2e9', linestyle='--', label='Baseline Aleatório (AUC = 0.5000)')
    plt.title('Curvas ROC (Receiver Operating Characteristic) Comparativas', color='#00f0ff', fontsize=13, pad=15)
    plt.xlabel('Taxa de Falsos Positivos (1 - Especificidade)', color='#e2e2e9')
    plt.ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)', color='#e2e2e9')
    plt.legend(facecolor='#1a1a1e', labelcolor='#e2e2e9', loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, "story_desempenho_roc.png"), dpi=300, facecolor='#121214')
    plt.close()
    
    # B. Curvas Precision-Recall (PR)
    plt.figure(figsize=(8, 7))
    baseline_pr = np.sum(y_sample) / len(y_sample)
    for name, p in probs.items():
        prec, rec, _ = precision_recall_curve(y_sample, p)
        plt.plot(rec, prec, lw=2, label=f'{name}')
    plt.axhline(y=baseline_pr, color='#ff007f', linestyle='--', label=f'Baseline Proporcional ({baseline_pr*100:.1f}%)')
    plt.title('Curvas Precision-Recall (PR) Comparativas', color='#00f0ff', fontsize=13, pad=15)
    plt.xlabel('Recall (Sensibilidade)', color='#e2e2e9')
    plt.ylabel('Precision (Precisão Preditiva)', color='#e2e2e9')
    plt.legend(facecolor='#1a1a1e', labelcolor='#e2e2e9', loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, "story_desempenho_pr.png"), dpi=300, facecolor='#121214')
    plt.close()
    
    # C. Curvas de Calibração (Reliability Diagrams)
    plt.figure(figsize=(8, 7))
    plt.plot([0, 1], [0, 1], 'k:', color='#e2e2e9', label='Calibração Perfeita')
    for name, p in probs.items():
        fraction_of_positives, mean_predicted_value = calibration_curve(y_sample, p, n_bins=10)
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', label=name, lw=2, markersize=4)
    plt.title('Reliability Diagram: Curvas de Calibração de Probabilidade', color='#00f0ff', fontsize=13, pad=15)
    plt.xlabel('Probabilidade Prevista Média', color='#e2e2e9')
    plt.ylabel('Proporção Real de Acidentes Graves', color='#e2e2e9')
    plt.legend(facecolor='#1a1a1e', labelcolor='#e2e2e9', loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, "story_desempenho_calibration.png"), dpi=300, facecolor='#121214')
    plt.close()
    
    # D. Curvas de Lift e Ganho Acumulado
    plot_lift_gain_curves(y_sample, probs, os.path.join(artifacts_dir, "story_desempenho_lift_gain.png"))
    
    # E. Fronteiras de Decisão 2D via t-SNE
    print("Rodando t-SNE nos atributos scaled para plot de fronteira de decisão...")
    tsne_indices = np.random.choice(len(X_sample_scaled), size=min(1200, len(X_sample_scaled)), replace=False)
    X_tsne_input = X_sample_scaled[tsne_indices]
    y_tsne_input = y_sample.iloc[tsne_indices]
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=600)
    X_tsne_2d = tsne.fit_transform(X_tsne_input)
    
    plt.figure(figsize=(9, 7))
    plt.scatter(X_tsne_2d[y_tsne_input == 0, 0], X_tsne_2d[y_tsne_input == 0, 1], 
                color='#00f0ff', alpha=0.6, label='Leve/Moderado (0)', s=20)
    plt.scatter(X_tsne_2d[y_tsne_input == 1, 0], X_tsne_2d[y_tsne_input == 1, 1], 
                color='#ff007f', alpha=0.6, label='Grave/Fatal (1)', s=20)
    plt.title('Projeção Espacial 2D via t-SNE: Separação de Classes de Severidade', color='#00f0ff', fontsize=12, pad=15)
    plt.xlabel('t-SNE Componente 1', color='#e2e2e9')
    plt.ylabel('t-SNE Componente 2', color='#e2e2e9')
    plt.legend(facecolor='#1a1a1e', labelcolor='#e2e2e9')
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, "story_desempenho_tsne.png"), dpi=300, facecolor='#121214')
    plt.close()
    
    # F. Explicações Globais com SHAP
    if SHAP_AVAILABLE:
        print("Calculando SHAP values para o XGBoost...")
        xgb_model = models['XGBoost']
        explainer = shap.TreeExplainer(xgb_model)
        
        # Amostra de 500 registros para o plot do SHAP
        shap_indices = np.random.choice(len(X_sample_scaled), size=500, replace=False)
        X_shap_input = X_sample_scaled[shap_indices]
        
        shap_values = explainer.shap_values(X_shap_input)
        
        plt.figure(figsize=(10, 6))
        # shap.summary_plot altera o plot atual
        shap.summary_plot(shap_values, X_sample_df.iloc[shap_indices], feature_names=feature_cols_eval, show=False)
        plt.title('Explicação Global SHAP (Tree SHAP no XGBoost)', color='#00f0ff', fontsize=13, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(artifacts_dir, "story_desempenho_shap.png"), dpi=300, facecolor='#121214')
        plt.close()
        print("SHAP Plot gerado com sucesso!")
        
    print("\nTodos os gráficos de storytelling gerados na pasta de artefatos com sucesso!")

if __name__ == "__main__":
    main()
