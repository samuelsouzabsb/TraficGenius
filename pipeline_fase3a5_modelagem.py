import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def get_chance_criteria(y):
    # Critérios de Chance
    proportions = y.value_counts(normalize=True)
    c_max = proportions.max() * 100
    c_prop = (proportions ** 2).sum() * 100
    return c_max, c_prop

def evaluate_model(name, model, X_test, y_test, c_max, c_prop):
    print(f"\n{'='*50}\nAvaliação do Modelo: {name}")
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    hit_ratio = accuracy_score(y_test, y_pred) * 100
    
    print("Matriz de Classificação (Confusão):")
    print(cm)
    print(f"\nRazão de Sucesso (Hit Ratio): {hit_ratio:.2f}%")
    print(f"Critério de Chance Máxima: {c_max:.2f}%")
    print(f"Critério de Chance Proporcional: {c_prop:.2f}%")
    
    if hit_ratio > c_max * 1.25: # Regra de ouro da Análise Multivariada (Hair et al.)
        print("-> [Aprovado] O modelo é substancialmente melhor que o Critério de Chance Máxima.")
    elif hit_ratio > c_prop * 1.25:
        print("-> [Aprovado] O modelo supera o Critério de Chance Proporcional com folga.")
    else:
        print("-> [Alerta] O modelo não apresenta poder preditivo muito superior ao acaso/distribuição natural.")
        
    print("\nRelatório Multiclasse:")
    print(classification_report(y_test, y_pred))

def run_pipeline():
    file_path = r"c:\Users\samuelbarroso\Documents\Desenvolvimento\TraficGenius\dataset\dataset_amostra_limpa.parquet"
    print("Carregando dados...")
    df = pd.read_parquet(file_path)
    
    # Fase 1/2 Cleanup
    # Removendo Wind_Chill(F) devido à altissima multicolinearidade detectada na Fase 2
    drop_cols = ['Wind_Chill(F)', 'Start_Time', 'Sunrise_Sunset'] 
    
    # Selecionando as numéricas e booleanas
    features = [c for c in df.columns if c not in drop_cols and c != 'Severity']
    X = df[features]
    y = df['Severity']
    
    # Normalização (Z-Score) crítica para Logit, MDA e Rede Neural
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fase 5: Validação Hold-out (Treino/Teste)
    print("\n--- Fase 5: Particionamento (Cross-Validation / Hold-Out) ---")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)
    print(f"Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")
    
    c_max, c_prop = get_chance_criteria(y_test)
    
    print("\n--- Fase 3 e 4: Treinamento de Modelos Concorrentes ---")
    
    # Modelo 1: Logit Ordenado
    # Na ausência do pacote `mord`, utilizamos LogisticRegression Multinomial
    # que é a aproximação padrão no Scikit-Learn para multiplas categorias.
    print("Treinando Modelo 1: Logit Multinomial/Ordinal...")
    logit = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    logit.fit(X_train, y_train)
    evaluate_model("Logit Multinomial (Proxy Ordinal)", logit, X_test, y_test, c_max, c_prop)
    
    # Pseudo R² McFadden simplificado: 1 - (LL_model / LL_null)
    # Como o sklearn não retorna log-likelihood diretamente de forma simples, baseamos-nos na acurácia acima.
    
    # Modelo 2: MDA (Análise Discriminante Múltipla)
    print("\nTreinando Modelo 2: Análise Discriminante Múltipla (MDA)...")
    mda = LinearDiscriminantAnalysis()
    mda.fit(X_train, y_train)
    evaluate_model("MDA (Linear Discriminant Analysis)", mda, X_test, y_test, c_max, c_prop)
    
    # Modelo 3: Random Forest (Não-Paramétrico)
    print("\nTreinando Modelo 3: Random Forest (CART)...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    evaluate_model("Random Forest", rf, X_test, y_test, c_max, c_prop)
    
    # Modelo 4: Rede Neural (MLP)
    print("\nTreinando Modelo 4: Rede Neural Artificial (MLP)...")
    mlp = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=200, random_state=42, early_stopping=True)
    mlp.fit(X_train, y_train)
    evaluate_model("Rede Neural (MLP)", mlp, X_test, y_test, c_max, c_prop)

    # Cross-validation extra para verificar Overfitting no Random Forest
    print("\n--- Verificação de Overfitting (Random Forest) ---")
    print("Acurácia no Treino:", rf.score(X_train, y_train) * 100)
    print("Acurácia no Teste :", rf.score(X_test, y_test) * 100)
    
if __name__ == "__main__":
    run_pipeline()
