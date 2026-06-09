# -*- coding: utf-8 -*-
"""
Script de Treinamento Geral (3 Classes):
Treina e persiste os 3 modelos de ML de 3 classes (XGBoost, Random Forest e Regressão Logística).
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(project_root, "dataset")
    file_path = os.path.join(dataset_dir, "dataset_amostra_limpa_avancado.parquet")
    
    print("Carregando dataset...")
    df = pd.read_parquet(file_path)
    print(f"Dataset carregado com {len(df):,} linhas.")
    
    drop_cols = ['Sensacao_Termica_F', 'Nascer_Por_Sol']
    features = [c for c in df.columns if c not in drop_cols and c != 'Severidade']
    
    X = df[features]
    
    # Mapeamento para 3 classes:
    # 1 e 2 -> 0 (Leve/Médio), 3 -> 1 (Grave), 4 -> 2 (Fatal)
    print("Mapeando target para 3 classes...")
    y = df['Severidade'].map({1: 0, 2: 0, 3: 1, 4: 2})
    
    print("Ajustando padronizador (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Salvando padronizador 3 classes...")
    joblib.dump(scaler, os.path.join(dataset_dir, "scaler_3classes.joblib"))
    
    print("Calculando pesos de classes...")
    sample_weights = compute_sample_weight(class_weight='balanced', y=y)
    
    # 1. XGBoost
    print("Treinando e salvando XGBoost 3 Classes...")
    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='mlogloss',
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_scaled, y, sample_weight=sample_weights)
    joblib.dump(xgb, os.path.join(dataset_dir, "xgboost_model_3classes.joblib"))
    
    # 2. Random Forest
    print("Treinando e salvando Random Forest 3 Classes...")
    rf = RandomForestClassifier(
        n_estimators=30,
        max_depth=6,
        max_samples=0.1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_scaled, y)
    joblib.dump(rf, os.path.join(dataset_dir, "random_forest_model_3classes.joblib"))
    
    # 3. Logistic Regression
    print("Treinando e salvando Regressão Logística 3 Classes...")
    lr = LogisticRegression(
        max_iter=100,
        solver='lbfgs',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    lr.fit(X_scaled, y)
    joblib.dump(lr, os.path.join(dataset_dir, "logistic_regression_model_3classes.joblib"))
    
    print("Todos os modelos de 3 classes foram treinados e persistidos com sucesso!")
