# -*- coding: utf-8 -*-
"""
Script de Diagnósticos Estatísticos Multivariados
Calcula testes de premissas clássicas, multicolinearidade, autocorrelação e outliers.
Salva todos os resultados em dataset/diagnosticos_estatisticos.json.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset

warnings.filterwarnings('ignore')

def get_continuous_features(df):
    # Seleciona features numéricas que representam grandezas físicas contínuas
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

def calculate_mahalanobis(df_continuous):
    # Calcula distância de Mahalanobis D^2 em matriz
    # Usa pseudo-inversa caso a covariância seja quase singular
    mean = df_continuous.mean(axis=0)
    cov = df_continuous.cov()
    inv_cov = np.linalg.pinv(cov)
    
    diff = df_continuous - mean
    d2 = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
    return d2

def run_runs_test(residuals):
    # Teste de runs (das carreiras) para independência dos erros
    median_val = np.median(residuals)
    runs = np.diff(residuals > median_val) != 0
    num_runs = np.sum(runs) + 1
    n1 = np.sum(residuals > median_val)
    n2 = np.sum(residuals <= median_val)
    
    expected_runs = ((2.0 * n1 * n2) / (n1 + n2)) + 1.0
    var_runs = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n1 - n2)) / (((n1 + n2) ** 2) * (n1 + n2 - 1.0))
    z_stat = (num_runs - expected_runs) / np.sqrt(var_runs)
    p_val = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_stat)))
    return z_stat, p_val

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(project_root, "dataset", "dataset_amostra_limpa_binaria.parquet")
    output_json = os.path.join(project_root, "dataset", "diagnosticos_estatisticos.json")
    
    print("[INFO] Carregando dataset...")
    df = pd.read_parquet(input_file)
    print(f"[INFO] Dataset carregado. Shape: {df.shape}")
    
    # Amostragem representativa para testes computationais
    np.random.seed(42)
    sample_size_large = min(50000, len(df))
    sample_size_small = min(5000, len(df))
    
    df_sample_large = df.sample(n=sample_size_large, random_state=42).copy()
    df_sample_small = df.sample(n=sample_size_small, random_state=42).copy()
    
    continuous_cols = get_continuous_features(df)
    
    results = {}
    
    # -------------------------------------------------------------
    # 1. Análise de Outliers
    # -------------------------------------------------------------
    print("\n--- Analisando Outliers ---")
    outlier_res = {}
    
    # Z-Score Univariado (Standard e Modified) para velocidade_media
    for col in ['velocidade_media', 'temperatura_celsius', 'precipitacao_milimetros']:
        if col in df.columns:
            series = df_sample_large[col]
            z_score = np.abs(stats.zscore(series))
            outlier_z = np.sum(z_score > 3.5) / len(series)
            
            # Modified Z-Score
            median = series.median()
            mad = np.median(np.abs(series - median))
            # Evita divisão por zero
            if mad == 0:
                mad = 1e-5
            mod_z = 0.6745 * (series - median) / mad
            outlier_mod_z = np.sum(np.abs(mod_z) > 3.5) / len(series)
            
            outlier_res[col] = {
                'outliers_z_percent': float(outlier_z * 100),
                'outliers_mod_z_percent': float(outlier_mod_z * 100),
                'mean': float(series.mean()),
                'median': float(median),
                'mad': float(mad)
            }
            
    # Mahalanobis D^2 Multivariado
    print("Calculando Distância de Mahalanobis D^2...")
    df_cont_large = df_sample_large[continuous_cols]
    d2 = calculate_mahalanobis(df_cont_large)
    df_deg = len(continuous_cols)
    crit_val = stats.chi2.ppf(0.999, df=df_deg) # p < 0.001
    mahalanobis_outliers = np.sum(d2 > crit_val) / len(d2)
    
    outlier_res['multivariate_mahalanobis'] = {
        'degrees_of_freedom': df_deg,
        'critical_value_999': float(crit_val),
        'max_d2': float(np.max(d2)),
        'mean_d2': float(np.mean(d2)),
        'outliers_percent': float(mahalanobis_outliers * 100)
    }
    results['outliers'] = outlier_res
    
    # -------------------------------------------------------------
    # Regressão Auxiliar para Ajuste de Resíduos
    # -------------------------------------------------------------
    print("\nAjustando Regressão Auxiliar para geração de resíduos...")
    # Prepara matrizes
    df_reg = df_sample_large.copy()
    df_reg['pais_US'] = (df_reg['pais'] == 'US').astype(int)
    
    # Trata categóricos (dummies)
    cat_cols = ['rodovia_dominante_res11', 'rodovia_dominante_res10', 'rodovia_dominante_res9']
    df_reg_encoded = pd.get_dummies(df_reg, columns=cat_cols, drop_first=True)
    
    target_col = 'severidade_binaria'
    feature_cols = [c for c in df_reg_encoded.columns if c not in [target_col, 'data_inversa', 'pais']]
    
    # Converte booleanos em int
    for c in feature_cols:
        if df_reg_encoded[c].dtype == bool:
            df_reg_encoded[c] = df_reg_encoded[c].astype(int)
            
    X_reg = sm.add_constant(df_reg_encoded[feature_cols].astype(float))
    y_reg = df_reg_encoded[target_col].astype(float)
    
    # Ajusta Mínimos Quadrados (OLS) para obter os resíduos do modelo linear clássico
    ols_model = sm.OLS(y_reg, X_reg).fit()
    residuals_large = ols_model.resid
    
    # Resíduos no sample pequeno para testes lentos
    df_reg_small = df_sample_small.copy()
    df_reg_small['pais_US'] = (df_reg_small['pais'] == 'US').astype(int)
    df_reg_small_encoded = pd.get_dummies(df_reg_small, columns=cat_cols, drop_first=True)
    
    # Garante mesmas colunas no sample pequeno
    for c in feature_cols:
        if c not in df_reg_small_encoded.columns:
            df_reg_small_encoded[c] = 0
            
    X_reg_small = sm.add_constant(df_reg_small_encoded[feature_cols].astype(float))
    y_reg_small = df_reg_small_encoded[target_col].astype(float)
    
    ols_model_small = sm.OLS(y_reg_small, X_reg_small).fit()
    residuals_small = ols_model_small.resid
    
    # -------------------------------------------------------------
    # 2. Testes de Normalidade
    # -------------------------------------------------------------
    print("\n--- Testes de Normalidade nos Resíduos ---")
    normality_res = {}
    
    # Assimetria e Curtose
    skewness = float(stats.skew(residuals_large))
    kurtosis = float(stats.kurtosis(residuals_large))
    
    # Jarque-Bera
    jb_stat, jb_pval = stats.jarque_bera(residuals_large)
    
    # Shapiro-Wilk (requer amostra menor)
    sw_stat, sw_pval = stats.shapiro(residuals_small)
    
    # Kolmogorov-Smirnov (resíduos padronizados vs. normal)
    std_residuals = (residuals_large - residuals_large.mean()) / residuals_large.std()
    ks_stat, ks_pval = stats.kstest(std_residuals, 'norm')
    
    normality_res = {
        'skewness': skewness,
        'kurtosis': kurtosis,
        'jarque_bera': {'statistic': float(jb_stat), 'p_value': float(jb_pval)},
        'shapiro_wilk': {'statistic': float(sw_stat), 'p_value': float(sw_pval)},
        'kolmogorov_smirnov': {'statistic': float(ks_stat), 'p_value': float(ks_pval)}
    }
    results['normality'] = normality_res
    
    # -------------------------------------------------------------
    # 3. Homocedasticity
    # -------------------------------------------------------------
    print("\n--- Testes de Homocedasticidade ---")
    homo_res = {}
    
    # Breusch-Pagan
    bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(residuals_large, X_reg)
    
    # Teste de Levene para velocidade_media com base na severidade (0 vs 1)
    grp0 = df_sample_large[df_sample_large['severidade_binaria'] == 0]['velocidade_media']
    grp1 = df_sample_large[df_sample_large['severidade_binaria'] == 1]['velocidade_media']
    levene_stat, levene_pval = stats.levene(grp0, grp1)
    
    homo_res = {
        'breusch_pagan': {
            'lm_statistic': float(bp_lm),
            'lm_p_value': float(bp_lm_p),
            'f_statistic': float(bp_f),
            'f_p_value': float(bp_f_p)
        },
        'levene_velocidade_media': {
            'statistic': float(levene_stat),
            'p_value': float(levene_pval)
        }
    }
    results['homoscedasticity'] = homo_res
    
    # -------------------------------------------------------------
    # 4. Independência dos Erros
    # -------------------------------------------------------------
    print("\n--- Testes de Independência dos Erros ---")
    ind_res = {}
    
    # Durbin-Watson
    dw_stat = sm.stats.durbin_watson(residuals_large)
    
    # Teste das Carreiras (Runs Test)
    runs_z, runs_pval = run_runs_test(residuals_large.to_numpy())
    
    ind_res = {
        'durbin_watson': float(dw_stat),
        'runs_test': {
            'statistic_z': float(runs_z),
            'p_value': float(runs_pval)
        }
    }
    results['independence'] = ind_res
    
    # -------------------------------------------------------------
    # 5. Multicolinearidade
    # -------------------------------------------------------------
    print("\n--- Testes de Multicolinearidade ---")
    multi_res = {}
    
    # VIF
    vifs = []
    # Calcula VIF para as features numéricas contínuas (VIF > 10 indica multicolinearidade severa)
    X_cont = df_sample_large[continuous_cols].astype(float)
    X_cont = sm.add_constant(X_cont)
    
    for i in range(X_cont.shape[1]):
        col_name = X_cont.columns[i]
        val = variance_inflation_factor(X_cont.values, i)
        vifs.append({'feature': col_name, 'vif': float(val)})
        
    # Condition Index (SVD do X padronizado)
    X_scaled = (X_cont.iloc[:, 1:] - X_cont.iloc[:, 1:].mean()) / X_cont.iloc[:, 1:].std()
    X_scaled = sm.add_constant(X_scaled).dropna(axis=1, how='all')
    U, s, Vt = np.linalg.svd(X_scaled)
    s_max = np.max(s)
    condition_indices = [float(s_max / val) for val in s if val > 0]
    
    multi_res = {
        'vif_values': vifs,
        'max_condition_index': float(np.max(condition_indices)),
        'condition_indices': sorted(condition_indices, reverse=True)
    }
    results['multicollinearity'] = multi_res
    
    # -------------------------------------------------------------
    # 6. RESET de Ramsey (Especificação do Modelo)
    # -------------------------------------------------------------
    print("\n--- Teste RESET de Ramsey ---")
    reset_res = {}
    
    # Ramsey RESET
    try:
        reset_test = linear_reset(ols_model_small, power=3, test_type='fitted')
        reset_res = {
            'statistic': float(reset_test.statistic),
            'p_value': float(reset_test.pvalue)
        }
    except Exception as e:
        reset_res = {'error': str(e)}
        
    results['specification_reset'] = reset_res
    
    # Grava JSON com todos os diagnósticos estatísticos
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n[INFO] Diagnósticos salvos com sucesso em {output_json}!")

if __name__ == "__main__":
    main()
