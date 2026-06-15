# -*- coding: utf-8 -*-
"""
Pipeline Fases 3 a 5: Modelagem Preditiva Avançada (Binária com Stacking e Threshold Tuning)
Treina e compara 5 modelos base: Regressão Logística, LDA, Random Forest, Rede Neural (MLP) e XGBoost.
Implementa Stacking Ensemble (Modelo 6) com meta-aprendedor Logit.
Otimiza o limiar de decisão de cada modelo para maximizar o F1-Score da classe grave.
Todos os dados usam nomenclatura em português.
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
from sklearn.model_selection import train_test_split, KFold, cross_val_score
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
    class_counts = y.value_counts()
    min_class_size = class_counts.min()
    half_size = min(size // 2, min_class_size)
    
    np.random.seed(42)
    indices_resampled = []
    for cls in class_counts.index:
        idx_cls = y[y == cls].index
        chosen = np.random.choice(idx_cls, size=half_size, replace=False)
        indices_resampled.extend(chosen)
        
    X_res = X.loc[indices_resampled]
    y_res = y.loc[indices_resampled]
    
    shuffle_idx = np.random.permutation(len(X_res))
    return X_res.iloc[shuffle_idx], y_res.iloc[shuffle_idx]

def mcfadden_pseudo_r2(y_true, y_prob):
    N = len(y_true)
    y_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)
    ll_fit = -N * log_loss(y_true, y_prob)
    p1 = np.mean(y_true)
    null_probs = np.full(N, p1)
    ll_null = -N * log_loss(y_true, null_probs)
    return 1.0 - (ll_fit / ll_null)

def calculate_discriminant_loadings(X_scaled, y, lda, feature_names):
    scores = lda.transform(X_scaled).ravel()
    loadings = []
    for i in range(X_scaled.shape[1]):
        corr = np.corrcoef(X_scaled[:, i], scores)[0, 1]
        loadings.append(corr)
    df_loadings = pd.DataFrame({
        'Feature': feature_names,
        'Carga_Discriminante': loadings
    })
    df_loadings['Carga_Absoluta'] = df_loadings['Carga_Discriminante'].abs()
    return df_loadings.sort_values(by='Carga_Absoluta', ascending=False)

def build_keras_mlp(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def find_best_threshold(y_true, y_prob):
    """
    Encontra o limiar de decisão ótimo que maximiza o F1-Score da classe Grave (1).
    """
    best_thresh = 0.5
    best_f1 = 0
    # Testa limiares de 0.05 a 0.95
    for thresh in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(project_root, "dataset", "dataset_amostra_limpa_binaria.parquet")
    features_json_path = os.path.join(project_root, "dataset", "features_selecionadas_binaria.json")
    output_dir = os.path.join(project_root, "dataset")
    
    print("Carregando base completa com variáveis de interação física e lista de features...")
    df = pd.read_parquet(dataset_path)
    with open(features_json_path, 'r', encoding='utf-8') as f:
        features = json.load(f)
        
    categorical_cols = ['rodovia_dominante_res11', 'rodovia_dominante_res10', 'rodovia_dominante_res9']
    
    print(f"Aplicando One-Hot Encoding nas colunas categóricas...")
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    del df
    gc.collect()
    
    target_col = 'severidade_binaria'
    feature_cols = [col for col in df_encoded.columns if col != target_col]
    
    X = df_encoded[feature_cols]
    y = df_encoded[target_col]
    
    # Divisão Treino/Teste (70/30) estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    print(f"Frequência de classe no Treino: {y_train.value_counts().to_dict()}")
    print(f"Frequência de classe no Teste: {y_test.value_counts().to_dict()}")
    
    # Divisão do Treino em Base (85%) e Meta (15%) para Stacking (evitar leakage) e Threshold Tuning
    X_base, X_meta, y_base, y_meta = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )
    
    # Padronização (Scaler) ajustado no conjunto de treinamento base
    scaler = StandardScaler()
    print("Ajustando o Scaler no conjunto de treinamento Base...")
    X_base_scaled = scaler.fit_transform(X_base)
    X_meta_scaled = scaler.transform(X_meta)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = scaler.transform(X_train) # Para ajuste final
    
    # Salva o Scaler
    joblib.dump(scaler, os.path.join(output_dir, "scaler_binaria.joblib"))
    
    # Dicionário de base estimators
    base_models = {}
    
    # -------------------------------------------------------------
    # 1. Regressão Logística
    # -------------------------------------------------------------
    print("\n--- Treinando Modelo 1: Regressão Logística (Logit) ---")
    logit = LogisticRegression(max_iter=150, solver='lbfgs', random_state=42, n_jobs=-1)
    logit.fit(X_base_scaled, y_base)
    base_models['Logistic Regression'] = logit
    
    # -------------------------------------------------------------
    # 2. LDA (Priors 50/50)
    # -------------------------------------------------------------
    print("\n--- Treinando Modelo 2: Análise Discriminante Linear (LDA) ---")
    lda = LinearDiscriminantAnalysis(priors=[0.5, 0.5])
    lda.fit(X_base_scaled, y_base)
    base_models['LDA'] = lda
    
    # -------------------------------------------------------------
    # 3. Random Forest (Otimizado)
    # -------------------------------------------------------------
    print("\n--- Treinando Modelo 3: Random Forest ---")
    rf = RandomForestClassifier(n_estimators=30, max_depth=12, class_weight='balanced', max_samples=0.10, random_state=42, n_jobs=-1)
    rf.fit(X_base_scaled, y_base)
    base_models['Random Forest'] = rf
    
    # -------------------------------------------------------------
    # 4. Rede Neural Artificial (MLP)
    # -------------------------------------------------------------
    print("\n--- Treinando Modelo 4: Rede Neural Artificial (MLP) ---")
    if TENSORFLOW_AVAILABLE:
        # Splits internos para early stopping no Keras
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_base_scaled, y_base, test_size=0.15, stratify=y_base, random_state=42
        )
        classes_unique = np.unique(y_tr)
        class_weights_vals = compute_class_weight(class_weight='balanced', classes=classes_unique, y=y_tr)
        class_weight_dict = dict(zip(classes_unique, class_weights_vals))
        
        mlp_model = build_keras_mlp(X_base_scaled.shape[1])
        early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        
        mlp_model.fit(
            X_tr, y_tr,
            epochs=8,
            batch_size=2048,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            class_weight=class_weight_dict,
            verbose=1
        )
        base_models['Neural Network (MLP)'] = mlp_model
        del X_tr, X_val, y_tr, y_val
        gc.collect()
    else:
        print("Subamostrando para MLPClassifier (fallback)...")
        X_tr_bal, y_tr_bal = get_balanced_subset(X_base, y_base, size=400000)
        X_tr_bal_scaled = scaler.transform(X_tr_bal)
        
        mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150, random_state=42, early_stopping=True)
        mlp_model.fit(X_tr_bal_scaled, y_tr_bal)
        base_models['Neural Network (MLP)'] = mlp_model
        del X_tr_bal, y_tr_bal, X_tr_bal_scaled
        gc.collect()
        
    # -------------------------------------------------------------
    # 5. XGBoost
    # -------------------------------------------------------------
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
    
    # -------------------------------------------------------------
    # Geração do Meta-Dataset (Stacking) e Otimização do Limiar
    # -------------------------------------------------------------
    print("\n--- Gerando Meta-Dataset para o Stacking (Holdout) ---")
    
    meta_train_preds = {}
    meta_test_preds = {}
    
    for name, model in base_models.items():
        print(f"Gerando predições para {name}...")
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
    
    # Treinamento do Meta-Learner (Logistic Regression)
    print("\n--- Treinando Modelo 6: Stacking Ensemble (Meta-Learner Logit) ---")
    meta_learner = LogisticRegression(max_iter=150, solver='lbfgs', random_state=42, n_jobs=-1)
    meta_learner.fit(X_meta_train, y_meta)
    
    # Salva todos os modelos
    joblib.dump(logit, os.path.join(output_dir, "model_logit_binaria.joblib"))
    joblib.dump(lda, os.path.join(output_dir, "model_lda_binaria.joblib"))
    joblib.dump(rf, os.path.join(output_dir, "model_rf_binaria.joblib"))
    joblib.dump(xgb, os.path.join(output_dir, "model_xgb_binaria.joblib"))
    joblib.dump(meta_learner, os.path.join(output_dir, "model_stacking_binaria.joblib"))
    if not TENSORFLOW_AVAILABLE:
        joblib.dump(mlp_model, os.path.join(output_dir, "model_mlp_binaria.joblib"))
    else:
        mlp_model.save(os.path.join(output_dir, "model_mlp_binaria.h5"))
        
    # Salva coeficientes do Stacking (Contribuição de cada modelo base)
    print("\nPesos das Contribuições no Stacking Ensemble:")
    for name, coef in zip(X_meta_train.columns, meta_learner.coef_[0]):
        print(f" - {name:20}: {coef:.4f}")
        
    # -------------------------------------------------------------
    # Otimização de Limiar de Decisão (Tuning Threshold)
    # -------------------------------------------------------------
    print("\n--- Otimizando Limiares de Decisão (Threshold Tuning no Meta-Treino) ---")
    
    optimal_thresholds = {}
    for name in X_meta_train.columns:
        p_meta = X_meta_train[name]
        thresh = find_best_threshold(y_meta, p_meta)
        optimal_thresholds[name] = thresh
        print(f" - Limiar ótimo para {name:20}: {thresh:.2f}")
        
    # Limiar do Stacking
    p_meta_stack = meta_learner.predict_proba(X_meta_train)[:, 1]
    optimal_thresholds['Stacking Ensemble'] = find_best_threshold(y_meta, p_meta_stack)
    print(f" - Limiar ótimo para Stacking Ensemble  : {optimal_thresholds['Stacking Ensemble']:.2f}")
    
    # Salva limiares de decisão
    with open(os.path.join(output_dir, "limiares_decisao_binaria.json"), 'w') as f:
        json.dump(optimal_thresholds, f, indent=4)
        
    # -------------------------------------------------------------
    # Avaliação Comparativa de Desempenho no Teste
    # -------------------------------------------------------------
    print("\n--- 4. Avaliação Comparativa de Desempenho no Teste ---")
    
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
        
        # Predições antes da otimização (Corte padrão de 50%)
        y_pred_default = (y_prob >= 0.5).astype(int)
        acc_def = accuracy_score(y_test, y_pred_default)
        bal_def = balanced_accuracy_score(y_test, y_pred_default)
        f1_def = f1_score(y_test, y_pred_default)
        
        # Predições após otimização (Corte no limiar ótimo)
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
        
        # Matriz de Confusão com Limiar Ótimo
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
        
        plt.title(f"Matriz Confusão (Limiar {opt_thresh:.2f}) - {name}", fontsize=11, color='#00f0ff', pad=15)
        plt.xlabel("Classificação Predita", fontsize=10, color='#e2e2e9')
        plt.ylabel("Classificação Real", fontsize=10, color='#e2e2e9')
        plt.xticks(color='#e2e2e9')
        plt.yticks(color='#e2e2e9')
        plt.tight_layout()
        
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(os.path.join(output_dir, f"matriz_confusao_{safe_name}.png"), dpi=300, facecolor='#121214')
        plt.close()
        
    metrics_df = pd.DataFrame(metrics_summary)
    print("\nTabela Comparativa de Métricas (Stacking e Threshold Tuning):")
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(os.path.join(output_dir, "comparativo_modelos_binaria.csv"), index=False)
    
    # Gráfico Comparativo Final
    plt.figure(figsize=(12, 6))
    x = np.arange(len(metrics_df))
    width = 0.35
    
    plt.bar(x - width/2, metrics_df['F1-Score Macro'] * 100, width, label='F1-Score Macro (%)', color='#ff007f')
    plt.bar(x + width/2, metrics_df['Acurácia Global (Ótimo)'] * 100, width, label='Acurácia Global (%)', color='#00f0ff')
    
    plt.axhline(y=pcc * 100, color='#e2e2e9', linestyle=':', label=f'PCC ({pcc*100:.1f}%)')
    plt.axhline(y=pcc_target_125 * 100, color='#ffcc00', linestyle='--', label=f'1.25 * PCC ({pcc_target_125*100:.1f}%)')
    plt.axhline(y=mcc * 100, color='#00ff66', linestyle='-.', label=f'MCC ({mcc*100:.1f}%)')
    
    plt.title("Comparativo dos Modelos com Stacking e Limiar Ótimo", fontsize=14, color='#00f0ff', pad=15)
    plt.ylabel("Pontuação (%)", fontsize=12, color='#e2e2e9')
    plt.xticks(x, metrics_df['Modelo'], color='#e2e2e9', rotation=15)
    plt.yticks(color='#e2e2e9')
    plt.legend(facecolor='#1a1a1e', edgecolor='#2d2d34', labelcolor='#e2e2e9', loc='lower right')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "comparativo_performance_binaria.png"), dpi=300, facecolor='#121214')
    plt.close()
    
    # Importância de Features XGBoost (Top 15)
    importances = xgb.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(15), importances[indices][::-1], color='#00f0ff', height=0.6)
    plt.yticks(range(15), [feature_cols[i] for i in indices][::-1], color='#e2e2e9')
    plt.xlabel("Importância Relativa (F-Score)", fontsize=12, color='#e2e2e9')
    plt.title("Importância das Variáveis (Top 15 - XGBoost)", fontsize=14, color='#00f0ff', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "importancia_features_xgboost_binaria.png"), dpi=300, facecolor='#121214')
    plt.close()
    
    # Texto listando as importâncias no console para visualização direta do usuário
    print("\nSorted Feature Importances (XGBoost):")
    sorted_importances = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    for f_name, imp_val in sorted_importances[:20]:
        print(f" - {f_name:40}: {imp_val:.6f}")
        
    print("\nFases 3 a 5 concluídas com Stacking, Threshold Tuning e Feature Importances!")
