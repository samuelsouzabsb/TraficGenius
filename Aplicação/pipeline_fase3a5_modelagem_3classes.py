# -*- coding: utf-8 -*-
"""
Pipeline Fases 3 a 5: Modelagem Preditiva Avançada de 3 Classes (XGBoost & Deep Learning CNN 1D)
Este script treina, sintoniza (tuning) e avalia os modelos preditivos para 3 classes de severidade:
0: Leve/Médio, 1: Grave, 2: Fatal.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Tenta carregar o TensorFlow de forma segura
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except (ImportError, Exception) as e:
    print(f"\nAviso: Nao foi possivel carregar o runtime do TensorFlow ({e}).")
    TENSORFLOW_AVAILABLE = False

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

def plot_smote_distribution_3classes(y_before, y_after, output_dir):
    apply_chart_style()
    print("Gerando gráfico comparativo do balanceamento (3 classes)...")
    
    # Mapeia 0-2 para G1-G3 para legibilidade do usuário
    y_b = y_before + 1
    counts_before = y_b.value_counts().sort_index()
    counts_after = pd.Series([len(y_before) // len(counts_before)] * len(counts_before), index=counts_before.index)
    
    labels = ["Leve/Médio", "Grave", "Fatal"]
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, counts_before.values, width, label='Antes do Balanceamento', color='#ff007f')
    rects2 = ax.bar(x + width/2, counts_after.values, width, label='Pesos de Classe Equivalentes', color='#00f0ff')
    
    ax.set_title('Impacto do Balanceamento por Pesos (3 Classes)', fontsize=14, color='#00f0ff', pad=15)
    ax.set_xlabel('Classes de Severidade', fontsize=12, color='#e2e2e9')
    ax.set_ylabel('Quantidade / Peso Relativo', fontsize=12, color='#e2e2e9')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color='#e2e2e9')
    ax.tick_params(colors='#e2e2e9')
    ax.legend(facecolor='#1a1a1e', edgecolor='#2d2d34', labelcolor='#e2e2e9')
    
    for rect in rects1 + rects2:
        height = int(rect.get_height())
        ax.annotate(f'{height:,}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, color='#e2e2e9')
                    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "distribuicao_classes_smote_3classes.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#121214')
    plt.close()
    print(f"Gráfico de distribuição salvo em: {out_path}")

def plot_feature_importance_3classes(model, features, output_dir):
    apply_chart_style()
    print("Gerando gráfico de importância de atributos do XGBoost (3 classes)...")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], color='#00f0ff', height=0.6)
    plt.yticks(range(len(indices)), [features[i] for i in indices], color='#e2e2e9')
    plt.xticks(color='#e2e2e9')
    plt.xlabel("Importância Relativa (F-Score)", fontsize=12, color='#e2e2e9')
    plt.title("Importância das Variáveis (XGBoost 3 Classes)", fontsize=14, color='#00f0ff', pad=15)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "importancia_features_xgboost_3classes.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#121214')
    plt.close()
    print(f"Gráfico de importância salvo em: {out_path}")

def plot_confusion_matrix_heatmap_3classes(y_test, y_pred, output_dir):
    apply_chart_style()
    print("Gerando heatmap da matriz de confusão (3 classes)...")
    
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Leve/Médio", "Grave", "Fatal"]
    
    plt.figure(figsize=(8, 6))
    cmap = sns.dark_palette('#ff007f', as_cmap=True)
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    annot_labels = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_labels[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)"
            
    sns.heatmap(cm, annot=annot_labels, fmt='', cmap=cmap, xticklabels=labels, yticklabels=labels,
                cbar=False, linewidths=0.5, linecolor='#121214')
    
    plt.title("Matriz de Confusão - XGBoost (3 Classes)", fontsize=14, color='#00f0ff', pad=15)
    plt.xlabel("Severidade Predita", fontsize=12, color='#e2e2e9')
    plt.ylabel("Severidade Real", fontsize=12, color='#e2e2e9')
    plt.xticks(color='#e2e2e9')
    plt.yticks(color='#e2e2e9')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "matriz_confusao_xgboost_3classes.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#121214')
    plt.close()
    print(f"Matriz de confusão salva em: {out_path}")

def plot_model_comparison_3classes(metrics_dict, output_dir):
    apply_chart_style()
    print("Gerando gráfico comparativo de performance (3 classes)...")
    
    models = list(metrics_dict.keys())
    accuracies = [metrics_dict[m]['accuracy'] for m in models]
    f1_scores = [metrics_dict[m]['f1_macro'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, f1_scores, width, label='F1-Score Macro (%)', color='#ff007f')
    rects2 = ax.bar(x + width/2, accuracies, width, label='Acurácia Global (%)', color='#00f0ff')
    
    ax.set_title('Comparativo de Desempenho (3 Classes)', fontsize=16, color='#00f0ff', pad=20)
    ax.set_ylabel('Pontuação (%)', fontsize=12, color='#e2e2e9')
    ax.set_xticks(x)
    ax.set_xticklabels(models, color='#e2e2e9', fontsize=11)
    ax.tick_params(colors='#e2e2e9')
    ax.set_ylim(0, 110)
    ax.legend(facecolor='#1a1a1e', edgecolor='#2d2d34', labelcolor='#e2e2e9')
    
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, color='#e2e2e9')
                    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "comparativo_performance_modelos_3classes.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#121214')
    plt.close()
    print(f"Gráfico comparativo salvo em: {out_path}")

def get_chance_criteria(y):
    proportions = y.value_counts(normalize=True)
    c_max = proportions.max() * 100
    c_prop = (proportions ** 2).sum() * 100
    return c_max, c_prop

def build_cnn1d_3classes(input_shape, num_classes):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, BatchNormalization
    
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(name, y_test, y_pred, y_prob=None, c_max=None, c_prop=None):
    print(f"\n{'='*50}\nAvaliação do Modelo: {name}")
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro') * 100
    
    print("Matriz de Classificação (Confusão):")
    print(cm)
    print(f"\nF1-Score Macro: {f1:.2f}%")
    
    if y_prob is not None:
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr') * 100
        print(f"ROC-AUC (OVR): {roc_auc:.2f}%")
        
    if c_max is not None and c_prop is not None:
        hit_ratio = (np.trace(cm) / np.sum(cm)) * 100
        print(f"Hit Ratio (Acurácia): {hit_ratio:.2f}% | C.Max: {c_max:.2f}% | C.Prop: {c_prop:.2f}%")
        
    print("\nRelatório Multiclasse:")
    print(classification_report(y_test, y_pred))
    
    hit_ratio = (np.trace(cm) / np.sum(cm)) * 100
    return {"accuracy": hit_ratio, "f1_macro": f1}

def run_pipeline():
    project_root = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(project_root, "dataset", "dataset_amostra_limpa_avancado.parquet")
    print("Carregando dados avançados...")
    df = pd.read_parquet(file_path)
    print(f"Dataset carregado com {len(df):,} registros.")
        
    output_dir = os.path.dirname(file_path)
    drop_cols = ['Sensacao_Termica_F', 'Nascer_Por_Sol']
    features = [c for c in df.columns if c not in drop_cols and c != 'Severidade']
    X = df[features]
    
    # Mapeamento para 3 classes:
    # 1 e 2 -> 0 (Leve/Médio), 3 -> 1 (Grave), 4 -> 2 (Fatal)
    print("Mapeando target para 3 classes de severidade...")
    y = df['Severidade'].map({1: 0, 2: 0, 3: 1, 4: 2})
    num_classes = 3
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\n--- Particionamento e Pesos de Classe (3 Classes) ---")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)
    
    from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
    unique_classes = np.unique(y_train)
    class_weights_vals = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
    class_weight_dict = dict(zip(unique_classes, class_weights_vals))
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    plot_smote_distribution_3classes(y_train, y_train, output_dir)
    c_max, c_prop = get_chance_criteria(y_test)
    
    # 1. XGBoost
    print("Treinando XGBoost Classifier (3 Classes)...")
    best_xgb = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='mlogloss',
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    best_xgb.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Salva modelos específicos de 3 classes
    model_save_path = os.path.join(output_dir, "xgboost_model_3classes.joblib")
    scaler_save_path = os.path.join(output_dir, "scaler_3classes.joblib")
    joblib.dump(best_xgb, model_save_path)
    joblib.dump(scaler, scaler_save_path)
    print(f"Modelo XGBoost 3 Classes salvo em: {model_save_path}")
    
    metrics_dict = {}
    y_pred_xgb = best_xgb.predict(X_test)
    y_prob_xgb = best_xgb.predict_proba(X_test)
    xgb_metrics = evaluate_model("XGBoost (3 Classes)", y_test, y_pred_xgb, y_prob_xgb, c_max, c_prop)
    metrics_dict["XGBoost"] = xgb_metrics
    
    plot_feature_importance_3classes(best_xgb, features, output_dir)
    plot_confusion_matrix_heatmap_3classes(y_test, y_pred_xgb, output_dir)
    
    # 2. CNN 1D
    if TENSORFLOW_AVAILABLE:
        from tensorflow.keras.callbacks import EarlyStopping
        print("\nTreinando CNN 1D (3 Classes)...")
        X_train_cnn = np.expand_dims(X_train, axis=2)
        X_test_cnn = np.expand_dims(X_test, axis=2)
        
        cnn_model = build_cnn1d_3classes(input_shape=(X_train_cnn.shape[1], 1), num_classes=num_classes)
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        cnn_model.fit(X_train_cnn, y_train, epochs=5, batch_size=2048, validation_split=0.2, 
                      callbacks=[early_stop], class_weight=class_weight_dict, verbose=1)
        
        y_prob_cnn = cnn_model.predict(X_test_cnn)
        y_pred_cnn = np.argmax(y_prob_cnn, axis=1)
        
        cnn_metrics = evaluate_model("CNN 1D (3 Classes)", y_test, y_pred_cnn, y_prob_cnn, c_max, c_prop)
        metrics_dict["CNN 1D"] = cnn_metrics
        
    # 3. Random Forest
    print("\nTreinando Random Forest (3 Classes)...")
    rf_model = RandomForestClassifier(n_estimators=30, max_depth=6, max_samples=0.1, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)
    rf_metrics = evaluate_model("Random Forest (3 Classes)", y_test, y_pred_rf, y_prob_rf, c_max, c_prop)
    metrics_dict["Random Forest"] = rf_metrics
    
    # 4. Regressão Logística
    print("\nTreinando Regressão Logística (3 Classes)...")
    lr_model = LogisticRegression(max_iter=100, solver='lbfgs', class_weight='balanced', random_state=42, n_jobs=-1)
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    y_prob_lr = lr_model.predict_proba(X_test)
    lr_metrics = evaluate_model("Regressão Logística (3 Classes)", y_test, y_pred_lr, y_prob_lr, c_max, c_prop)
    metrics_dict["Regressão Logística"] = lr_metrics
    
    plot_model_comparison_3classes(metrics_dict, output_dir)
    print("\nFase de Modelagem de 3 Classes concluída!")

if __name__ == "__main__":
    run_pipeline()
