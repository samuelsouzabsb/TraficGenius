# -*- coding: utf-8 -*-
"""
Pipeline Fases 3 a 5: Modelagem Preditiva Avançada (XGBoost & Deep Learning CNN 1D)
Este script treina, sintoniza (tuning) e avalia dois modelos preditivos robustos
para prever a gravidade de incidentes rodoviários (Severity Level - multiclasse 1 a 4).
1. XGBoost: Modelo baseado em árvores com aumento de gradiente (Gradient Boosted Trees).
2. CNN 1D (Convolutional Neural Network): Rede neural convolucional profunda para capturar padrões numéricos sequenciais.

Dicas de Inglês (English Tips):
- 'Oversampling' significa superamostragem (criar dados sintéticos para equilibrar as classes).
- 'SMOTE' (Synthetic Minority Over-sampling Technique) é a técnica de superamostragem de minorias sintéticas.
- 'Stratify/Stratified' refere-se ao particionamento de dados mantendo a proporção de cada classe constante nos conjuntos de treino e teste.
- 'Tuning/Hyperparameter tuning' é o ajuste/otimização fina dos hiperparâmetros de um modelo.
- 'Early Stopping' significa parada antecipada (interrompe o treino da rede neural se ela parar de melhorar).
- 'One-Versus-Rest (OVR)' é a técnica de avaliação de modelos multiclasse comparando cada classe contra todas as outras combinadas.
- 'Chance criteria' são os critérios de classificação ao acaso, servindo como a baseline mínima de acurácia que o modelo deve bater.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Desliga avisos e mensagens informativas do TensorFlow para simplificar os logs
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

def get_chance_criteria(y):
    """
    Calcula os critérios de classificação ao acaso para estabelecer baselines comparativas:
    1. Maximum Chance Criterion (C.Max): Acurácia caso sempre chutassemos a classe majoritária.
    2. Proportional Chance Criterion (C.Prop): Acurácia esperada considerando as frequências reais de todas as classes.
    
    Parâmetros (Parameters):
    - y (pd.Series): As classes reais observadas (ground truth labels).
    
    Retorno (Returns):
    - c_max (float): Percentual limite do C.Max.
    - c_prop (float): Percentual limite do C.Prop.
    """
    proportions = y.value_counts(normalize=True)  # Frequência relativa de cada classe
    c_max = proportions.max() * 100
    c_prop = (proportions ** 2).sum() * 100
    return c_max, c_prop

def build_cnn1d_model(input_shape, num_classes):
    """
    Constrói a arquitetura de uma rede neural convolucional unidimensional (CNN 1D) sequencial.
    Focada em extrair padrões locais nas features numéricas padronizadas.
    
    Parâmetros (Parameters):
    - input_shape (tuple): Dimensão do formato de entrada (ex: (número de features, 1)).
    - num_classes (int): Quantidade de classes na variável alvo (Severity categories).
    
    Retorno (Returns):
    - model (tf.keras.Model): Modelo Keras compilado.
    """
    model = Sequential([
        # Primeira camada convolucional: extrai 64 filtros de tamanho 3
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        # Normalização em lote para acelerar a convergência e dar estabilidade numérica ao treino
        BatchNormalization(),
        # Max Pooling: reduz o tamanho dimensional pela metade, retendo a característica mais forte do mapa de ativação
        MaxPooling1D(pool_size=2),
        # Dropout: desliga aleatoriamente 20% dos neurônios nesta etapa para prevenir sobreajuste (overfitting)
        Dropout(0.2),
        
        # Segunda camada convolucional profunda: extrai 128 filtros
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Converte a matriz tridimensional resultante em um vetor plano unidimensional
        Flatten(),
        # Camada densamente conectada (Fully Connected Layer)
        Dense(64, activation='relu'),
        Dropout(0.3),
        # Camada final de saída com ativação Softmax para estimar a distribuição de probabilidades das classes
        Dense(num_classes, activation='softmax')
    ])
    
    # Compilação utilizando o otimizador Adam e perda sparse categorical crossentropy para labels inteiros
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(name, y_test, y_pred, y_prob=None, c_max=None, c_prop=None):
    """
    Gera um relatório de avaliação detalhado com as métricas de performance do modelo preditivo.
    Apresenta a matriz de confusão, F1-Score, ROC-AUC e testa se bateu as baselines ao acaso.
    
    Parâmetros (Parameters):
    - name (str): Nome amigável do modelo avaliado.
    - y_test (array): Classes de teste verdadeiras.
    - y_pred (array): Classes de teste preditas.
    - y_prob (array, opcional): Probabilidades estimadas para cada classe.
    - c_max (float, opcional): C.Max calculado para comparação.
    - c_prop (float, opcional): C.Prop calculado para comparação.
    """
    print(f"\n{'='*50}\nAvaliação do Modelo: {name}")
    
    # Matriz de confusão (Confusion Matrix)
    cm = confusion_matrix(y_test, y_pred)
    # F1-Score Macro: Média aritmética dos F1-scores de cada classe (imparcial ao desbalanceamento)
    f1 = f1_score(y_test, y_pred, average='macro') * 100
    
    print("Matriz de Classificação (Confusão):")
    print(cm)
    print(f"\nF1-Score Macro: {f1:.2f}%")
    
    # ROC-AUC (Área sob a curva característica de operação do receptor)
    if y_prob is not None:
        # Multi-classe One-versus-Rest (OVR) avalia a curva ROC de cada classe contra o restante
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr') * 100
        print(f"ROC-AUC (OVR): {roc_auc:.2f}%")
        
    # Valida o desempenho obtido contra os critérios do acaso (Chance baseline test)
    if c_max is not None and c_prop is not None:
        hit_ratio = (np.trace(cm) / np.sum(cm)) * 100  # Acurácia global obtida
        print(f"Hit Ratio (Acurácia): {hit_ratio:.2f}% | C.Max: {c_max:.2f}% | C.Prop: {c_prop:.2f}%")
        
    print("\nRelatório Multiclasse:")
    # classification_report calcula precisão, recall, f1-score e suporte por classe
    print(classification_report(y_test, y_pred))

def run_pipeline():
    """
    Função principal do pipeline de modelagem:
    - Carrega a base limpa gerada na Fase 1.
    - Remove colunas desnecessárias ou com VIF severo.
    - Padroniza variáveis numéricas (Standardization).
    - Divide em conjuntos de treino e teste mantendo a distribuição de classes (Stratification).
    - Aplica o SMOTE no treino para balancear a base de forma sintética.
    - Treina e ajusta hiperparâmetros do XGBoost.
    - Treina a Rede Neural Convolucional 1D (CNN 1D).
    """
    file_path = r"c:\Users\samuelbarroso\Documents\Desenvolvimento\TraficGenius\dataset\dataset_amostra_limpa_avancado.parquet"
    print("Carregando dados avançados...")
    df = pd.read_parquet(file_path)
    
    # Colunas a serem descartadas: VIF extremo (Wind_Chill(F)) e variáveis não numéricas (Sunrise_Sunset)
    drop_cols = ['Wind_Chill(F)', 'Sunrise_Sunset'] 
    
    features = [c for c in df.columns if c not in drop_cols and c != 'Severity']
    X = df[features]
    
    # Ajusta o label para iniciar de 0 (Exigência do algoritmo XGBoost multiclasse)
    # y mapeia as severidades {1, 2, 3, 4} para {0, 1, 2, 3}
    y = df['Severity'] - 1 
    num_classes = len(np.unique(y))
    
    # Padronização de variáveis contínuas (Média 0 e Variância 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\n--- Fase 5: Particionamento e SMOTE ---")
    # Particionamento Estratificado (stratify=y) garante que treino e teste tenham proporções idênticas de severidade
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)
    
    # Balanceamento sintético via SMOTE apenas no treino para não induzir vazamento de dados (data leakage) no teste
    print("Aplicando SMOTE no conjunto de treino...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Treino Original: {X_train.shape[0]} | Treino SMOTE: {X_train_res.shape[0]}")
    
    c_max, c_prop = get_chance_criteria(y_test)
    
    print("\n--- Fase 3 e 4: Treinamento de Modelos ---")
    
    # --- Modelo 1: XGBoost Classifier com Sintonia de Hiperparâmetros (Hyperparameter Tuning) ---
    print("Treinando XGBoost Classifier com Tuning...")
    xgb_base = XGBClassifier(eval_metric='mlogloss', random_state=42)
    
    # Espaço de busca para os hiperparâmetros
    param_dist = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200]
    }
    
    # Randomized Search executa um número controlado de iterações sorteando do espaço paramétrico.
    # Usamos n_iter=3 com validação cruzada 3-fold (cv=3) para otimizar o tempo de processamento.
    xgb_search = RandomizedSearchCV(xgb_base, param_dist, n_iter=3, cv=3, scoring='f1_macro', n_jobs=-1, random_state=42)
    xgb_search.fit(X_train_res, y_train_res)
    
    best_xgb = xgb_search.best_estimator_
    print(f"Melhores hiperparâmetros XGBoost: {xgb_search.best_params_}")
    
    # Prediz as severidades e as probabilidades de cada registro do teste
    y_pred_xgb = best_xgb.predict(X_test)
    y_prob_xgb = best_xgb.predict_proba(X_test)
    evaluate_model("XGBoost (Tuned)", y_test, y_pred_xgb, y_prob_xgb, c_max, c_prop)
    
    # --- Modelo 2: CNN 1D (Deep Learning via TensorFlow) ---
    print("\nTreinando Convolutional Neural Network (CNN 1D)...")
    # Redimensiona o input bidimensional (samples, features) para tridimensional (samples, features, channels=1)
    # exigência obrigatória para as convoluções 1D atuarem.
    X_train_cnn = np.expand_dims(X_train_res, axis=2)
    X_test_cnn = np.expand_dims(X_test, axis=2)
    
    cnn_model = build_cnn1d_model(input_shape=(X_train_cnn.shape[1], 1), num_classes=num_classes)
    
    # Callback de Early Stopping: aborta o treinamento se a perda de validação (val_loss) parar de diminuir
    # por 3 épocas consecutivas (patience=3), restaurando a melhor rodada de pesos encontrada.
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # Ajuste dos pesos (fit) com tamanho de lote de 256 e época limite 10
    cnn_model.fit(X_train_cnn, y_train_res, epochs=10, batch_size=256, validation_split=0.2, callbacks=[early_stop], verbose=1)
    
    # Calcula as probabilidades estimadas pela rede convolucional
    y_prob_cnn = cnn_model.predict(X_test_cnn)
    # Extrai o índice do neurônio de maior ativação (argmax) como a classe predita
    y_pred_cnn = np.argmax(y_prob_cnn, axis=1)
    
    evaluate_model("CNN 1D (Keras/TensorFlow)", y_test, y_pred_cnn, y_prob_cnn, c_max, c_prop)
    
if __name__ == "__main__":
    run_pipeline()
