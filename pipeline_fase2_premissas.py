import pandas as pd
import numpy as np
from scipy.stats import shapiro, levene
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore')

def test_normality_homoscedasticity(df):
    print("--- 1. Testes de Normalidade e Homocedasticidade ---")
    num_cols = ['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']
    
    # Amostra menor para testes estatísticos, pois Shapiro-Wilk não é confiável para N > 5000
    # e P-valores tendem a zero em amostras muito grandes.
    df_sample = df.sample(n=4000, random_state=42)
    
    print(f"Testando Normalidade (Shapiro-Wilk) em amostra de N=4000:")
    for col in num_cols:
        stat, p = shapiro(df_sample[col])
        is_normal = "Sim" if p > 0.05 else "Nao (Assimetria/Curtose)"
        print(f" - {col:20}: P-Valor={p:.4f} | Normal? {is_normal}")
        
    print("\nTestando Homocedasticidade (Teste de Levene entre Severidade 2 e 3, maiores grupos):")
    # Grupos para o teste de Levene
    group_2 = df[df['Severity'] == 2]
    group_3 = df[df['Severity'] == 3]
    
    for col in num_cols:
        stat, p = levene(group_2[col], group_3[col])
        is_homosc = "Sim" if p > 0.05 else "Nao (Variancias diferentes)"
        print(f" - {col:20}: P-Valor={p:.4f} | Homocedastico? {is_homosc}")
        
    print("\n[Conclusão Parcial]: Em Big Data, a normalidade quase sempre é rejeitada. Modelos como Random Forest e Redes Neurais são imunes a isso. O Logit Ordinal também não exige normalidade estrita das independentes, mas sim ausência de multicolinearidade perfeita.")

def check_multicollinearity(df):
    print("\n--- 2. Analise de Multicolinearidade (VIF e Tolerância) ---")
    num_cols = ['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']
    
    X = df[num_cols].dropna()
    
    vif_list = []
    # Calculo manual do VIF: 1 / (1 - R2)
    for col in num_cols:
        y_vif = X[col]
        X_vif = X.drop(columns=[col])
        
        model = LinearRegression()
        model.fit(X_vif, y_vif)
        r_sq = model.score(X_vif, y_vif)
        
        # Evitar divisão por zero caso haja colinearidade perfeita
        vif = 1 / (1 - r_sq) if r_sq < 1 else float('inf')
        vif_list.append(vif)
        
    vif_data = pd.DataFrame({
        "feature": num_cols,
        "VIF": vif_list
    })
    vif_data["Tolerancia"] = 1 / vif_data["VIF"]
    
    print(vif_data.sort_values(by="VIF", ascending=False).to_string(index=False))
    
    # Critério gerencial
    high_vif = vif_data[vif_data["VIF"] > 5]
    if not high_vif.empty:
        print("\n[ATENÇÃO] Variaveis com VIF > 5 (Possível redundância e risco para o Logit Ordinal):")
        print(high_vif["feature"].tolist())
        print("Recomendação: Para a modelagem paramétrica (Logit/MDA), vamos remover a redundância.")
    else:
        print("\n[OK] Não há sinais severos de Multicolinearidade.")

if __name__ == "__main__":
    print("Iniciando Fase 2...")
    file_path = r"c:\Users\samuelbarroso\Documents\Desenvolvimento\TraficGenius\dataset\dataset_amostra_limpa.parquet"
    print("Lendo parquet...")
    df = pd.read_parquet(file_path)
    print(f"Dataset carregado. Shape: {df.shape}")
    
    test_normality_homoscedasticity(df)
    check_multicollinearity(df)
    print("Fase 2 finalizada!")
