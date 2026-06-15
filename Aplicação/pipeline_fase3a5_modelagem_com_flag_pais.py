# -*- coding: utf-8 -*-
"""
Pipeline Fases 3 a 5: Modelagem Preditiva com Flag de País
Treina e compara os 5 classificadores e Stacking Ensemble usando o dataset unificado, 
incluindo a flag de país (pais_US = 1 para EUA, 0 para Brasil) como variável preditora.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, f1_score,
    accuracy_score, balanced_accuracy_score, log_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Tenta carregar o TensorFlow de forma segura
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
    print("[INFO] TensorFlow carregado com sucesso. Rede Neural usará o Keras.")
except (ImportError, Exception) as e:
    print(f"[AVISO] Não foi possível carregar o TensorFlow ({e}).")
    print("A Rede Neural usará o MLPClassifier do scikit-learn como fallback.")
    TENSORFLOW_AVAILABLE = False
    from sklearn.neural_network import MLPClassifier

warnings.filterwarnings('ignore')
if TENSORFLOW_AVAILABLE:
    tf.get_logger().setLevel('ERROR')

def apply_chart_style():
    plt.style.use('dark_background')
    plt.rcParams['figure.facecolor'] = '#121214'
    plt.rcParams['axes.facecolor'] = '#1a1a1e'
    plt.rcParams['grid.color'] = '#2d2d34'
    plt.rcParams['font.sans-serif'] = 'sans-serif'
    plt.rcParams['font.family'] = 'sans-serif'

def get_balanced_subset(X, y, size=300000):
    """Retorna uma subamostra balanceada para acelerar o fallback do MLPClassifier"""
    y0_idx = y[y == 0].index
    y1_idx = y[y == 1].index
    half_size = min(size // 2, len(y1_idx))
    
    y0_sampled = y0_idx[np.random.choice(len(y0_idx), half_size, replace=False)]
    y1_sampled = y1_idx[np.random.choice(len(y1_idx), half_size, replace=False)]
    
    idx_sampled = np.concatenate([y0_sampled, y1_sampled])
    np.random.shuffle(idx_sampled)
    
    return X.loc[idx_sampled], y.loc[idx_sampled]

def mcfadden_pseudo_r2(y_true, y_prob):
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)
    ll_model = np.sum(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    
    p_null = np.mean(y_true)
    ll_null = len(y_true) * (p_null * np.log(p_null) + (1 - p_null) * np.log(1 - p_null))
    
    return 1 - (ll_model / ll_null)

def find_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.05, 0.95, 91)
    best_thresh = 0.5
    best_f1 = 0
    for t in thresholds:
        f1 = f1_score(y_true, (y_prob >= t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh

def build_keras_mlp(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(project_root, "dataset", "dataset_amostra_limpa_binaria.parquet")
    output_dir = os.path.join(project_root, "dataset")
    
    print("Carregando base limpa...")
    df = pd.read_parquet(input_file)
    print(f"Shape original: {df.shape}")
    
    # -------------------------------------------------------------
    # Pré-processamento e Criação da Flag de País
    # -------------------------------------------------------------
    print("\n--- Criando Variável Preditora de Flag de País ---")
    df['pais_US'] = (df['pais'] == 'US').astype(int)
    
    # A coluna pais é removida mas a flag de país criada (pais_US) é mantida
    df = df.drop(columns=['pais'])
    
    target_col = 'severidade_binaria'
    cat_cols = ['rodovia_dominante_res11', 'rodovia_dominante_res10', 'rodovia_dominante_res9']
    
    print("Aplicando One-Hot Encoding nas colunas categóricas...")
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Libera memória
    del df
    gc.collect()
    
    feature_cols = [col for col in df_encoded.columns if col != target_col]
    X = df_encoded[feature_cols]
    y = df_encoded[target_col]
    
    # Divisão Treino/Teste (70/30) estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    # Divisão do Treino em Base (85%) e Meta (15%) para Stacking
    X_base, X_meta, y_base, y_meta = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )
    
    # Padronização
    scaler = StandardScaler()
    print("Ajustando o Scaler no conjunto de treinamento Base...")
    X_base_scaled = scaler.fit_transform(X_base)
    X_meta_scaled = scaler.transform(X_meta)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = scaler.transform(X_train)
    
    joblib.dump(scaler, os.path.join(output_dir, "scaler_flag_pais.joblib"))
    
    base_models = {}
    
    # 1. Regressão Logística
    print("\n--- Treinando Modelo 1: Regressão Logística (Logit) ---")
    logit = LogisticRegression(max_iter=150, solver='lbfgs', random_state=42, n_jobs=-1)
    logit.fit(X_base_scaled, y_base)
    base_models['Logistic Regression'] = logit
    
    p_val_logit = logit.predict_proba(X_meta_scaled)[:, 1]
    pseudo_r2 = mcfadden_pseudo_r2(y_meta, p_val_logit)
    print(f"Regressão Logística concluída. Pseudo R^2: {pseudo_r2:.4f}")
    
    # 2. LDA
    print("\n--- Treinando Modelo 2: Análise Discriminante Linear (LDA) ---")
    lda = LinearDiscriminantAnalysis(priors=[0.5, 0.5])
    lda.fit(X_base_scaled, y_base)
    base_models['LDA'] = lda
    
    # 3. Random Forest
    print("\n--- Treinando Modelo 3: Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=30,
        max_depth=12,
        max_samples=0.10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_base_scaled, y_base)
    base_models['Random Forest'] = rf
    
    # 4. Rede Neural
    print("\n--- Treinando Modelo 4: Rede Neural Artificial (MLP) ---")
    if TENSORFLOW_AVAILABLE:
        mlp_model = build_keras_mlp(X_base.shape[1])
        class_weights = compute_class_weight('balanced', classes=np.unique(y_base), y=y_base)
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        
        mlp_model.fit(
            X_base_scaled, y_base,
            epochs=8,
            batch_size=4096,
            validation_data=(X_meta_scaled, y_meta),
            class_weight=class_weight_dict,
            callbacks=[early_stopping],
            verbose=1
        )
        base_models['Neural Network (MLP)'] = mlp_model
    else:
        print("Subamostrando para MLPClassifier (fallback)...")
        X_tr_bal, y_tr_bal = get_balanced_subset(X_base, y_base, size=400000)
        X_tr_bal_scaled = scaler.transform(X_tr_bal)
        mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150, random_state=42, early_stopping=True)
        mlp_model.fit(X_tr_bal_scaled, y_tr_bal)
        base_models['Neural Network (MLP)'] = mlp_model
        del X_tr_bal, y_tr_bal, X_tr_bal_scaled
        gc.collect()
        
    # 5. XGBoost
    print("\n--- Treinando Modelo 5: XGBoost ---")
    ratio = float(y_base.value_counts()[0]) / y_base.value_counts()[1]
    xgb = XGBClassifier(
        n_estimators=80,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=ratio,
        eval_metric='logloss',
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_base_scaled, y_base)
    base_models['XGBoost'] = xgb
    
    # Stacking
    print("\n--- Gerando Meta-Dataset para o Stacking (Holdout) ---")
    meta_train_preds = {}
    meta_test_preds = {}
    
    for name, model in base_models.items():
        if name == 'Neural Network (MLP)' and TENSORFLOW_AVAILABLE:
            p_meta = model.predict(X_meta_scaled).ravel()
            p_test = model.predict(X_test_scaled).ravel()
        else:
            p_meta = model.predict_proba(X_meta_scaled)[:, 1]
            p_test = model.predict_proba(X_test_scaled)[:, 1]
            
        meta_train_preds[name] = p_meta
        meta_test_preds[name] = p_test
        
    X_meta_train = pd.DataFrame(meta_train_preds)
    X_meta_test = pd.DataFrame(meta_test_preds)
    
    print("\n--- Treinando Modelo 6: Stacking Ensemble (Meta-Learner Logit) ---")
    meta_learner = LogisticRegression(max_iter=150, solver='lbfgs', random_state=42, n_jobs=-1)
    meta_learner.fit(X_meta_train, y_meta)
    
    # Salva os modelos
    joblib.dump(logit, os.path.join(output_dir, "model_logit_flag_pais.joblib"))
    joblib.dump(lda, os.path.join(output_dir, "model_lda_flag_pais.joblib"))
    joblib.dump(rf, os.path.join(output_dir, "model_rf_flag_pais.joblib"))
    joblib.dump(xgb, os.path.join(output_dir, "model_xgb_flag_pais.joblib"))
    joblib.dump(meta_learner, os.path.join(output_dir, "model_stacking_flag_pais.joblib"))
    if not TENSORFLOW_AVAILABLE:
        joblib.dump(mlp_model, os.path.join(output_dir, "model_mlp_flag_pais.joblib"))
    else:
        mlp_model.save(os.path.join(output_dir, "model_mlp_flag_pais.h5"))
        
    print("\nPesos das Contribuições no Stacking (com Flag de País):")
    for name, coef in zip(X_meta_train.columns, meta_learner.coef_[0]):
        print(f" - {name:20}: {coef:.4f}")
        
    # Threshold Tuning
    print("\n--- Otimizando Limiares de Decisão (Threshold Tuning no Meta-Treino) ---")
    optimal_thresholds = {}
    for name in X_meta_train.columns:
        p_meta = X_meta_train[name]
        thresh = find_best_threshold(y_meta, p_meta)
        optimal_thresholds[name] = thresh
        print(f" - Limiar ótimo para {name:20}: {thresh:.2f}")
        
    p_meta_stack = meta_learner.predict_proba(X_meta_train)[:, 1]
    optimal_thresholds['Stacking Ensemble'] = find_best_threshold(y_meta, p_meta_stack)
    print(f" - Limiar ótimo para Stacking Ensemble  : {optimal_thresholds['Stacking Ensemble']:.2f}")
    
    with open(os.path.join(output_dir, "limiares_decisao_flag_pais.json"), 'w') as f:
        json.dump(optimal_thresholds, f, indent=4)
        
    # Avaliação no Teste
    print("\n--- Avaliação de Desempenho no Teste ---")
    p1 = np.mean(y_test)
    p0 = 1 - p1
    mcc = max(p0, p1)
    pcc = p0**2 + p1**2
    pcc_target_125 = 1.25 * pcc
    
    print(f"Proporção de Acidentes Leves/Moderados (0) no Teste: {p0*100:.2f}%")
    print(f"Proporção de Acidentes Graves/Fatais (1) no Teste: {p1*100:.2f}%")
    print(f"MCC: {mcc*100:.2f}% | PCC: {pcc*100:.2f}% | Meta (1.25*PCC): {pcc_target_125*100:.2f}%")
    
    all_eval_probs = X_meta_test.copy()
    all_eval_probs['Stacking Ensemble'] = meta_learner.predict_proba(X_meta_test)[:, 1]
    
    metrics_summary = []
    apply_chart_style()
    
    for name in all_eval_probs.columns:
        y_prob = all_eval_probs[name].to_numpy()
        
        # Padrão 50%
        y_pred_default = (y_prob >= 0.5).astype(int)
        acc_def = accuracy_score(y_test, y_pred_default)
        
        # Ótimo
        opt_thresh = optimal_thresholds[name]
        y_pred_opt = (y_prob >= opt_thresh).astype(int)
        
        acc_opt = accuracy_score(y_test, y_pred_opt)
        bal_opt = balanced_accuracy_score(y_test, y_pred_opt)
        f1_opt = f1_score(y_test, y_pred_opt)
        f1_macro_opt = f1_score(y_test, y_pred_opt, average='macro')
        roc_auc = roc_auc_score(y_test, y_prob)
        
        metrics_summary.append({
            'Modelo': name,
            'Limiar Ótimo': opt_thresh,
            'Acurácia Global (50%)': acc_def,
            'Acurácia Global (Ótimo)': acc_opt,
            'Acurácia Balanceada': bal_opt,
            'F1-Score Classe 1': f1_opt,
            'F1-Score Macro': f1_macro_opt,
            'ROC-AUC': roc_auc
        })
        
        # Matriz de Confusão
        cm = confusion_matrix(y_test, y_pred_opt)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(7, 5))
        cmap = sns.dark_palette('#ff007f', as_cmap=True)
        annot_labels = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot_labels[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)"
                
        sns.heatmap(cm, annot=annot_labels, fmt='', cmap=cmap, 
                    xticklabels=["Leve/Moderado", "Grave/Fatal"],
                    yticklabels=["Leve/Moderado", "Grave/Fatal"],
                    cbar=False, linewidths=0.5, linecolor='#121214')
        
        plt.title(f"Matriz Confusão (Limiar {opt_thresh:.2f}) - {name} (Flag País)", fontsize=11, color='#00f0ff', pad=15)
        plt.xlabel("Classificação Predita", fontsize=10, color='#e2e2e9')
        plt.ylabel("Classificação Real", fontsize=10, color='#e2e2e9')
        plt.xticks(color='#e2e2e9')
        plt.yticks(color='#e2e2e9')
        plt.tight_layout()
        
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(os.path.join(output_dir, f"matriz_confusao_{safe_name}_flag_pais.png"), dpi=300, facecolor='#121214')
        plt.close()
        
    metrics_df = pd.DataFrame(metrics_summary)
    print("\nTabela Comparativa de Métricas (com Flag de País):")
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(os.path.join(output_dir, "comparativo_modelos_flag_pais.csv"), index=False)
    
    # Gráfico Comparativo
    plt.figure(figsize=(12, 6))
    x_indices = np.arange(len(metrics_df))
    width = 0.35
    
    plt.bar(x_indices - width/2, metrics_df['F1-Score Macro'] * 100, width, label='F1-Score Macro (%)', color='#ff007f')
    plt.bar(x_indices + width/2, metrics_df['Acurácia Global (Ótimo)'] * 100, width, label='Acurácia Global (%)', color='#00f0ff')
    plt.axhline(y=pcc * 100, color='#e2e2e9', linestyle=':', label=f'PCC ({pcc*100:.1f}%)')
    plt.axhline(y=pcc_target_125 * 100, color='#ffcc00', linestyle='--', label=f'Meta 1.25*PCC ({pcc_target_125*100:.1f}%)')
    
    plt.title('Comparativo de Modelos Preditivos (com Flag de País)', fontsize=14, color='#00f0ff', pad=15)
    plt.xlabel('Algoritmos / Ensembles', color='#e2e2e9', fontsize=12)
    plt.ylabel('Desempenho (%)', color='#e2e2e9', fontsize=12)
    plt.xticks(x_indices, metrics_df['Modelo'], rotation=15, color='#e2e2e9')
    plt.yticks(color='#e2e2e9')
    plt.legend(facecolor='#1a1a1e', edgecolor='#2d2d34', labelcolor='#e2e2e9')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "comparativo_performance_flag_pais.png"), dpi=300, facecolor='#121214')
    plt.close()
    
    # Importância de Features XGBoost
    importances = xgb.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(15), importances[indices][::-1], color='#00f0ff', height=0.6)
    plt.yticks(range(15), [feature_cols[i] for i in indices][::-1], color='#e2e2e9')
    plt.xlabel("Importância Relativa (F-Score)", fontsize=12, color='#e2e2e9')
    plt.title("Importância das Variáveis (Top 15 - XGBoost com Flag de País)", fontsize=14, color='#00f0ff', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "importancia_features_xgboost_flag_pais.png"), dpi=300, facecolor='#121214')
    plt.close()
    
    print("\nSorted Feature Importances (com Flag de País):")
    sorted_importances = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)[:20]
    for feat, imp in sorted_importances:
        print(f" - {feat:40}: {imp:.6f}")
        
    print("\nPipeline de modelagem com Flag de País concluído com sucesso!")

if __name__ == "__main__":
    main()
