# -*- coding: utf-8 -*-
"""
Script de Avaliação Detalhada dos Modelos (3 Classes)
Calcula Matriz de Confusão 3x3, F1 por classe, Recall por classe e ROC-AUC (OVR, Macro, Weighted).
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Importação de proteção para o TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except (ImportError, Exception) as e:
    print(f"\nAviso: Nao foi possivel carregar o runtime do TensorFlow ({e}).")
    TENSORFLOW_AVAILABLE = False

def build_cnn1d_3classes(input_shape, num_classes):
    if not TENSORFLOW_AVAILABLE:
        return None
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

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(project_root, "dataset", "dataset_amostra_limpa_avancado.parquet")
    
    print("Carregando o dataset limpo para avaliação (3 classes)...")
    df = pd.read_parquet(dataset_path)
    
    print("Amostrando 200.000 registros para avaliação rápida e robusta (3 classes)...")
    df_sample = df.sample(n=200000, random_state=42).copy()
    
    drop_cols = ['Sensacao_Termica_F', 'Nascer_Por_Sol']
    features = [c for c in df_sample.columns if c not in drop_cols and c != 'Severidade']
    
    X = df_sample[features]
    
    # Mapeamento para 3 classes:
    # 1 e 2 -> 0 (Leve/Médio), 3 -> 1 (Grave), 4 -> 2 (Fatal)
    y = df_sample['Severidade'].map({1: 0, 2: 0, 3: 1, 4: 2})
    num_classes = 3
    
    # Split 70% treino / 30% teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    # Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Pesos
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    unique_classes = np.unique(y_train)
    class_weights_vals = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
    class_weight_dict = dict(zip(unique_classes, class_weights_vals))
    
    # 1. XGBoost
    print("\n--- Treinando XGBoost 3 Classes ---")
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric='mlogloss',
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    y_pred_xgb = xgb.predict(X_test_scaled)
    y_prob_xgb = xgb.predict_proba(X_test_scaled)
    
    # 2. Random Forest
    print("\n--- Treinando Random Forest 3 Classes ---")
    rf = RandomForestClassifier(
        n_estimators=30,
        max_depth=6,
        max_samples=0.1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    y_prob_rf = rf.predict_proba(X_test_scaled)
    
    # 3. Regressão Logística
    print("\n--- Treinando Regressão Logística 3 Classes ---")
    lr = LogisticRegression(
        max_iter=100,
        solver='lbfgs',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_prob_lr = lr.predict_proba(X_test_scaled)
    
    # 4. CNN 1D
    if TENSORFLOW_AVAILABLE:
        print("\n--- Treinando CNN 1D 3 Classes ---")
        X_train_cnn = np.expand_dims(X_train_scaled, axis=2)
        X_test_cnn = np.expand_dims(X_test_scaled, axis=2)
        
        cnn = build_cnn1d_3classes(input_shape=(X_train_cnn.shape[1], 1), num_classes=num_classes)
        early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        cnn.fit(X_train_cnn, y_train, epochs=5, batch_size=1024, validation_split=0.2, 
                callbacks=[early_stop], class_weight=class_weight_dict, verbose=1)
        
        y_prob_cnn = cnn.predict(X_test_cnn)
        y_pred_cnn = np.argmax(y_prob_cnn, axis=1)
        
        all_runs = {
            "XGBoost": (y_pred_xgb, y_prob_xgb),
            "Random Forest": (y_pred_rf, y_prob_rf),
            "Regressão Logística": (y_pred_lr, y_prob_lr),
            "CNN 1D": (y_pred_cnn, y_prob_cnn)
        }
    else:
        print("\n--- CNN 1D ignorada devido a erro de runtime do TensorFlow ---")
        all_runs = {
            "XGBoost": (y_pred_xgb, y_prob_xgb),
            "Random Forest": (y_pred_rf, y_prob_rf),
            "Regressão Logística": (y_pred_lr, y_prob_lr)
        }
    
    # Salvando em arquivo Markdown de saída na pasta do projeto
    report_lines = []
    report_lines.append("# Relatório Estatístico Detalhado das Métricas dos Modelos (3 Classes)\n")
    report_lines.append("Este relatório descreve detalhadamente o desempenho de cada um dos modelos treinados com 3 classes de severidade (Leve/Médio, Grave e Fatal).\n")
    
    class_names = ["Leve/Médio", "Grave", "Fatal"]
    
    for model_name, (y_pred, y_prob) in all_runs.items():
        print(f"\nCalculando métricas para {model_name}...")
        
        # Matriz de Confusão 3x3
        cm = confusion_matrix(y_test, y_pred)
        
        # Relatório de Classificação
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        
        # ROC-AUC por Classe (One-vs-Rest)
        roc_auc_per_class = roc_auc_score(y_test, y_prob, multi_class='ovr', average=None)
        
        # ROC-AUC Macro e Weighted
        roc_auc_macro = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
        roc_auc_weighted = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
        
        # Gerando seções do relatório
        report_lines.append(f"## Modelo: {model_name}\n")
        
        # 1. Matriz de Confusão
        report_lines.append("### 1. Matriz de Confusão")
        report_lines.append("A matriz mostra as predições nas colunas e os valores reais nas linhas:\n")
        report_lines.append("| Real \\ Predito | Leve/Médio | Grave | Fatal |")
        report_lines.append("| :--- | :---: | :---: | :---: |")
        for idx, row in enumerate(cm):
            report_lines.append(f"| **{class_names[idx]}** | {row[0]:,} | {row[1]:,} | {row[2]:,} |")
        report_lines.append("\n")
        
        # 2 e 3. F1 e Recall por Classe
        report_lines.append("### 2 e 3. Métricas por Classe (F1-Score e Recall)")
        report_lines.append("| Classe de Severidade | Recall (%) | F1-Score (%) |")
        report_lines.append("| :--- | :---: | :---: |")
        for cls_idx in range(3):
            cls_str = str(cls_idx)
            rec = report_dict[cls_str]['recall'] * 100
            f1 = report_dict[cls_str]['f1-score'] * 100
            report_lines.append(f"| **{class_names[cls_idx]}** | {rec:.2f}% | {f1:.2f}% |")
        report_lines.append("\n")
        
        # 4. Tipo de ROC-AUC
        report_lines.append("### 4. Valores de ROC-AUC (One-vs-Rest)")
        report_lines.append("| Tipo de ROC-AUC (OVR) | Valor (%) |")
        report_lines.append("| :--- | :---: |")
        for cls_idx in range(3):
            val = roc_auc_per_class[cls_idx] * 100
            report_lines.append(f"| **{class_names[cls_idx]} vs Rest** | {val:.2f}% |")
        report_lines.append(f"| **Macro ROC-AUC (Média Simples)** | **{roc_auc_macro * 100:.2f}%** |")
        report_lines.append(f"| **Weighted ROC-AUC (Média Ponderada por Suporte)** | **{roc_auc_weighted * 100:.2f}%** |")
        report_lines.append("\n---\n")
        
    report_content = "\n".join(report_lines)
    
    # Salvar relatório no projeto
    output_report_path = os.path.join(project_root, "relatorio_metricas_detalhadas_3classes.md")
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"\nRelatório gerado e salvo com sucesso em: {output_report_path}")
