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
# Tenta carregar o TensorFlow de forma segura para tratamento de incompatibilidades do sistema
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except (ImportError, Exception) as e:
    print(f"\nAviso: Nao foi possivel carregar o runtime do TensorFlow ({e}).")
    print("A modelagem com rede neural CNN 1D sera pulada nesta execucao. Prosseguindo com o XGBoost.\n")
    TENSORFLOW_AVAILABLE = False
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import warnings
import matplotlib.pyplot as plt
import seaborn as sns




# Desliga avisos e mensagens informativas do TensorFlow para simplificar os logs
warnings.filterwarnings('ignore')
if TENSORFLOW_AVAILABLE:
    tf.get_logger().setLevel('ERROR')

def apply_chart_style():
    """
    Configura o matplotlib/seaborn com a identidade visual neon escura do TraficGenius.
    """
    plt.style.use('dark_background')
    plt.rcParams['figure.facecolor'] = '#121214'
    plt.rcParams['axes.facecolor'] = '#1a1a1e'
    plt.rcParams['grid.color'] = '#2d2d34'
    plt.rcParams['font.sans-serif'] = 'sans-serif'
    plt.rcParams['font.family'] = 'sans-serif'

def plot_smote_distribution(y_before, y_after, output_dir):
    """
    Gera um gráfico de barras agrupadas comparando as distribuições de classes antes e depois do balanceamento (Pesos de Classe).
    """
    apply_chart_style()
    print("Gerando gráfico comparativo do balanceamento por pesos...")
    
    # Mapeia 0-3 para 1-4 para legibilidade do usuário (Severidade G1 a G4)
    y_b = y_before + 1
    
    counts_before = y_b.value_counts().sort_index()
    # Simula distribuição após balanceamento de pesos para fins visuais (onde todas as classes têm a mesma representatividade)
    counts_after = pd.Series([len(y_before) // len(counts_before)] * len(counts_before), index=counts_before.index)
    
    labels = [f"G{i}" for i in counts_before.index]
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, counts_before.values, width, label='Antes do Balanceamento (Desbalanceado)', color='#ff007f')
    rects2 = ax.bar(x + width/2, counts_after.values, width, label='Pesos de Classe Equivalentes (Balanceado)', color='#00f0ff')
    
    ax.set_title('Impacto do Balanceamento por Pesos de Classe', fontsize=14, color='#00f0ff', pad=15)
    ax.set_xlabel('Classes de Severidade', fontsize=12, color='#e2e2e9')
    ax.set_ylabel('Quantidade / Peso Relativo', fontsize=12, color='#e2e2e9')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color='#e2e2e9')
    ax.tick_params(colors='#e2e2e9')
    ax.legend(facecolor='#1a1a1e', edgecolor='#2d2d34', labelcolor='#e2e2e9')
    
    # Adiciona rótulos numéricos sobre as barras
    for rect in rects1 + rects2:
        height = int(rect.get_height())
        ax.annotate(f'{height:,}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, color='#e2e2e9')
                    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "distribuicao_classes_smote.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#121214')
    plt.close()
    print(f"Gráfico de distribuição salvo em: {out_path}")

def plot_feature_importance(model, features, output_dir):
    """
    Gera um gráfico de barras horizontal mostrando a importância relativa dos atributos.
    """
    apply_chart_style()
    print("Gerando gráfico de importância de atributos do XGBoost...")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], color='#00f0ff', height=0.6)
    plt.yticks(range(len(indices)), [features[i] for i in indices], color='#e2e2e9')
    plt.xticks(color='#e2e2e9')
    plt.xlabel("Importância Relativa (F-Score)", fontsize=12, color='#e2e2e9')
    plt.title("Importância das Variáveis no Modelo XGBoost", fontsize=14, color='#00f0ff', pad=15)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "importancia_features_xgboost.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#121214')
    plt.close()
    print(f"Gráfico de importância de features salvo em: {out_path}")

def plot_confusion_matrix_heatmap(y_test, y_pred, output_dir):
    """
    Gera um heatmap detalhado da matriz de confusão com anotações absolutas e percentuais.
    """
    apply_chart_style()
    print("Gerando heatmap da matriz de confusão...")
    
    cm = confusion_matrix(y_test, y_pred)
    labels = [f"G{i}" for i in range(1, len(np.unique(y_test))+1)]
    
    plt.figure(figsize=(8, 6))
    
    # Paleta de tons de rosa neon e cinza escuro
    cmap = sns.dark_palette('#ff007f', as_cmap=True)
    
    # Normalização para proporções
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Strings formatadas
    annot_labels = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_labels[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)"
            
    sns.heatmap(cm, annot=annot_labels, fmt='', cmap=cmap, xticklabels=labels, yticklabels=labels,
                cbar=False, linewidths=0.5, linecolor='#121214')
    
    plt.title("Matriz de Confusão - Classificador XGBoost", fontsize=14, color='#00f0ff', pad=15)
    plt.xlabel("Severidade Predita", fontsize=12, color='#e2e2e9')
    plt.ylabel("Severidade Real", fontsize=12, color='#e2e2e9')
    plt.xticks(color='#e2e2e9')
    plt.yticks(color='#e2e2e9')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "matriz_confusao_xgboost.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#121214')
    plt.close()
    print(f"Gráfico da matriz de confusão salvo em: {out_path}")

def plot_model_comparison(metrics_dict, output_dir):
    """
    Gera um gráfico de barras comparativo de acurácia e F1-Score Macro de todos os modelos.
    """
    apply_chart_style()
    print("Gerando gráfico comparativo de performance dos modelos...")
    
    models = list(metrics_dict.keys())
    accuracies = [metrics_dict[m]['accuracy'] for m in models]
    f1_scores = [metrics_dict[m]['f1_macro'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Rosa neon para F1-Score Macro, Ciano neon para Acurácia
    rects1 = ax.bar(x - width/2, f1_scores, width, label='F1-Score Macro (%)', color='#ff007f')
    rects2 = ax.bar(x + width/2, accuracies, width, label='Acurácia Global (%)', color='#00f0ff')
    
    ax.set_title('Comparativo de Desempenho dos Modelos Preditivos', fontsize=16, color='#00f0ff', pad=20)
    ax.set_ylabel('Pontuação (%)', fontsize=12, color='#e2e2e9')
    ax.set_xticks(x)
    ax.set_xticklabels(models, color='#e2e2e9', fontsize=11)
    ax.tick_params(colors='#e2e2e9')
    ax.set_ylim(0, 110)
    ax.legend(facecolor='#1a1a1e', edgecolor='#2d2d34', labelcolor='#e2e2e9')
    
    # Adiciona rótulos sobre as barras
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, color='#e2e2e9')
                    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "comparativo_performance_modelos.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#121214')
    plt.close()
    print(f"Gráfico comparativo de modelos salvo em: {out_path}")

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
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, BatchNormalization
    
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
    
    hit_ratio = (np.trace(cm) / np.sum(cm)) * 100
    return {"accuracy": hit_ratio, "f1_macro": f1}

def run_pipeline():
    """
    Função principal do pipeline de modelagem:
    - Carrega a base limpa gerada na Fase 1.
    - Remove colunas desnecessárias ou com VIF severo.
    - Padroniza variáveis numéricas (Standardization).
    - Divide em conjuntos de treino e teste mantendo a distribuição de classes (Stratification).
    - Treina e ajusta hiperparâmetros do XGBoost.
    - Treina a Rede Neural Convolucional 1D (CNN 1D).
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(project_root, "dataset", "dataset_amostra_limpa_avancado.parquet")
    print("Carregando dados avançados...")
    df = pd.read_parquet(file_path)
    print(f"Dataset carregado com {len(df):,} registros.")
        
    output_dir = os.path.dirname(file_path)
    
    # Colunas a serem descartadas: VIF extremo (Sensacao_Termica_F) e variáveis não numéricas (Nascer_Por_Sol)
    drop_cols = ['Sensacao_Termica_F', 'Nascer_Por_Sol'] 
    
    features = [c for c in df.columns if c not in drop_cols and c != 'Severidade']
    X = df[features]
    
    # Ajusta o label para iniciar de 0 (Exigência do algoritmo XGBoost multiclasse)
    # y mapeia as severidades {1, 2, 3, 4} para {0, 1, 2, 3}
    y = df['Severidade'] - 1 
    num_classes = len(np.unique(y))
    
    # Padronização de variáveis contínuas (Média 0 e Variância 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\n--- Fase 5: Particionamento e Pesos de Classe ---")
    # Particionamento Estratificado (stratify=y) garante que treino e teste tenham proporções idênticas de severidade
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)
    
    # Balanceamento via pesos de classe no treino (sem inflar os dados em memória)
    print("Calculando pesos de classes para balanceamento...")
    from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
    
    # Computa pesos das classes
    unique_classes = np.unique(y_train)
    class_weights_vals = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
    class_weight_dict = dict(zip(unique_classes, class_weights_vals))
    print(f"Pesos de classe calculados: {class_weight_dict}")
    
    # Cria pesos individuais por amostra (usado no XGBoost)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    X_train_res, y_train_res = X_train, y_train
    print(f"Treino Original: {X_train.shape[0]} | Treino Otimizado com Pesos: {X_train_res.shape[0]}")
    
    # Gera o gráfico de comparação das classes
    plot_smote_distribution(y_train, y_train_res, output_dir)
    
    c_max, c_prop = get_chance_criteria(y_test)
    
    print("\n--- Fase 3 e 4: Treinamento de Modelos ---")
    
    # --- Modelo 1: XGBoost Classifier (Otimizado Direct Fit) ---
    print("Treinando XGBoost Classifier com hiperparâmetros pré-definidos...")
    # tree_method='hist' é crucial para rodar de forma ultra-rápida em bases gigantescas
    best_xgb = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='mlogloss',
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    best_xgb.fit(X_train_res, y_train_res, sample_weight=sample_weights)
    
    # Salva o modelo treinado e o padronizador (scaler) em disco para uso do SaaS
    import joblib
    model_save_path = os.path.join(output_dir, "xgboost_model.joblib")
    scaler_save_path = os.path.join(output_dir, "scaler.joblib")
    print(f"Salvando modelo XGBoost em: {model_save_path}")
    joblib.dump(best_xgb, model_save_path)
    print(f"Salvando padronizador (scaler) em: {scaler_save_path}")
    joblib.dump(scaler, scaler_save_path)
    
    metrics_dict = {}
 
    # Prediz as severidades e as probabilidades de cada registro do teste
    y_pred_xgb = best_xgb.predict(X_test)
    y_prob_xgb = best_xgb.predict_proba(X_test)
    xgb_metrics = evaluate_model("XGBoost (Otimizado)", y_test, y_pred_xgb, y_prob_xgb, c_max, c_prop)
    metrics_dict["XGBoost (Otimizado)"] = xgb_metrics
    
    # Gera gráficos de avaliação para o XGBoost (Importância das Features e Matriz de Confusão)
    plot_feature_importance(best_xgb, features, output_dir)
    plot_confusion_matrix_heatmap(y_test, y_pred_xgb, output_dir)
    
    # --- Modelo 2: CNN 1D (Deep Learning via TensorFlow) ---
    if TENSORFLOW_AVAILABLE:
        from tensorflow.keras.callbacks import EarlyStopping
        print("\nTreinando Convolutional Neural Network (CNN 1D)...")
        # Redimensiona o input bidimensional (samples, features) para tridimensional (samples, features, channels=1)
        # exigência obrigatória para as convoluções 1D atuarem.
        X_train_cnn = np.expand_dims(X_train_res, axis=2)
        X_test_cnn = np.expand_dims(X_test, axis=2)
        
        cnn_model = build_cnn1d_model(input_shape=(X_train_cnn.shape[1], 1), num_classes=num_classes)
        
        # Callback de Early Stopping: aborta o treinamento se a perda de validação (val_loss) parar de diminuir
        # por 3 épocas consecutivas (patience=3), restaurando a melhor rodada de pesos encontrada.
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        # Ajuste dos pesos (fit) com tamanho de lote de 2048 e época limite 5 para rapidez no CPU
        cnn_model.fit(X_train_cnn, y_train_res, epochs=5, batch_size=2048, validation_split=0.2, callbacks=[early_stop], class_weight=class_weight_dict, verbose=1)
        
        # Calcula as probabilidades estimadas pela rede convolucional
        y_prob_cnn = cnn_model.predict(X_test_cnn)
        # Extrai o índice do neurônio de maior ativação (argmax) como a classe predita
        y_pred_cnn = np.argmax(y_prob_cnn, axis=1)
        
        cnn_metrics = evaluate_model("CNN 1D (Keras/TensorFlow)", y_test, y_pred_cnn, y_prob_cnn, c_max, c_prop)
        metrics_dict["CNN 1D"] = cnn_metrics
    else:
        print("\nTreinamento de CNN 1D pulado: TensorFlow nao esta disponivel no sistema.")
 
    # --- Modelo 3: Random Forest Classifier ---
    print("\nTreinando Random Forest Classifier...")
    # Otimizado com max_samples=0.1 para treinar na base massiva em segundos em vez de horas
    rf_model = RandomForestClassifier(n_estimators=30, max_depth=6, max_samples=0.1, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_model.fit(X_train_res, y_train_res)
    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)
    rf_metrics = evaluate_model("Random Forest", y_test, y_pred_rf, y_prob_rf, c_max, c_prop)
    metrics_dict["Random Forest"] = rf_metrics
 
    # --- Modelo 4: Regressão Logística ---
    print("\nTreinando Regressão Logística...")
    lr_model = LogisticRegression(max_iter=100, solver='lbfgs', class_weight='balanced', random_state=42, n_jobs=-1)
    lr_model.fit(X_train_res, y_train_res)
    y_pred_lr = lr_model.predict(X_test)
    y_prob_lr = lr_model.predict_proba(X_test)
    lr_metrics = evaluate_model("Regressão Logística", y_test, y_pred_lr, y_prob_lr, c_max, c_prop)
    metrics_dict["Regressão Logística"] = lr_metrics
 
    # --- Comparativo Final de Desempenho ---
    plot_model_comparison(metrics_dict, output_dir)
    
if __name__ == "__main__":
    run_pipeline()
