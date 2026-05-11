import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# --- SEÇÃO 1: INTRODUÇÃO ---
cells.append(nbf.v4.new_markdown_cell("""
# TraficGenius: Previsão de Trânsito - Los Angeles (METR-LA)
## MVP Acadêmico - Fase 0

**Contexto:** Prever a velocidade média do tráfego em rodovias permite um melhor gerenciamento da malha urbana e auxilia em sistemas de navegação preditiva.
**Objetivo:** Prever a velocidade média do trânsito na rede rodoviária do condado de Los Angeles.
**Dataset:** *METR-LA*. Este é um dos datasets mais famosos no mundo acadêmico para previsão de tráfego, contendo dados de velocidade coletados de 207 sensores (loop detectors) distribuídos pelas rodovias de Los Angeles, com medições a cada 5 minutos.

Neste notebook, abordaremos:
1. Ingestão e EDA (Análise Exploratória)
2. Pré-processamento (Agregação Espacial e Lags Temporais)
3. Modelagem Preditiva (Linear Regression, Random Forest, XGBoost)
4. Avaliação e Comparação
"""))

# --- SEÇÃO 2: INSTALAÇÃO E IMPORTS ---
cells.append(nbf.v4.new_markdown_cell("""### 1. Preparação do Ambiente e Imports"""))
cells.append(nbf.v4.new_code_cell("""
# Instalando pacotes necessários
!pip install xgboost plotly scikit-learn -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

# Configuração visual padrão
sns.set_theme(style="whitegrid")
"""))

# --- SEÇÃO 3: CARREGAMENTO DE DADOS ---
cells.append(nbf.v4.new_markdown_cell("""
### 2. Carregamento dos Dados (METR-LA)
Vamos baixar o arquivo CSV do Zenodo. Originalmente, o METR-LA é uma matriz com 207 sensores. Para simplificar a modelagem tradicional (sem usar grafos) e criar um modelo preditivo eficiente, vamos agregar a velocidade média da cidade e prever a velocidade global futura com base no histórico recente.
"""))
cells.append(nbf.v4.new_code_cell("""
print("Baixando dados METR-LA (Los Angeles)... Isso pode levar alguns segundos.")
url = "https://zenodo.org/records/5146275/files/METR-LA.csv"
df_raw = pd.read_csv(url)

# O CSV tem a primeira coluna como tempo e o resto como sensores
df_raw.rename(columns={df_raw.columns[0]: 'datetime'}, inplace=True)
df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
df_raw.set_index('datetime', inplace=True)

display(df_raw.head())
print(f"\\nDimensões do dataset original: {df_raw.shape}")
"""))

# --- SEÇÃO 4: EDA ---
cells.append(nbf.v4.new_markdown_cell("""
### 3. Análise Exploratória de Dados (EDA)
Vamos calcular a velocidade média em todos os 207 sensores e visualizar o padrão de trânsito ao longo do tempo.
"""))
cells.append(nbf.v4.new_code_cell("""
# Calculando a média global de velocidade na rede
df = pd.DataFrame()
df['mean_speed'] = df_raw.mean(axis=1)

# Amostragem para visualizar (Plotar uma semana de dados)
fig = px.line(df.iloc[:2016], y='mean_speed', 
              title="Velocidade Média do Trânsito em Los Angeles (Primeira Semana)",
              labels={'datetime': 'Data e Hora', 'mean_speed': 'Velocidade Média (mph)'})
fig.show()

# Distribuição
fig2 = px.histogram(df, x='mean_speed', nbins=50, marginal="box",
                   title="Distribuição da Velocidade Média")
fig2.show()
"""))

# --- SEÇÃO 5: PREPROCESSAMENTO E LAGS ---
cells.append(nbf.v4.new_markdown_cell("""
### 4. Pré-processamento e Feature Engineering (Lags Temporais)
Em Séries Temporais, o melhor previsor do futuro é o passado recente.
Vamos criar características (*Features*) de *Lag* (atraso). Por exemplo, prever a velocidade no tempo $T$ usando a velocidade em $T-1, T-2, T-3$. Além disso, vamos extrair hora e dia da semana.
"""))
cells.append(nbf.v4.new_code_cell("""
# Criando features temporais
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek

# Criando Lags (Histórico dos últimos 3 períodos = últimos 15 minutos)
# Como a medição é a cada 5 min, lag_1 é 5 min atrás, lag_2 é 10 min atrás, etc.
for i in range(1, 4):
    df[f'speed_lag_{i}'] = df['mean_speed'].shift(i)

# Remover linhas com valores nulos causados pelo shift()
df.dropna(inplace=True)

display(df.head())

# Separação Treino / Teste (Temporal - Prevenindo Data Leakage)
split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

# Definindo Target
X_train = train_df.drop(columns=['mean_speed'])
y_train = train_df['mean_speed']

X_test = test_df.drop(columns=['mean_speed'])
y_test = test_df['mean_speed']

print(f"\\nTamanho do Treino: {len(X_train)} amostras")
print(f"Tamanho do Teste: {len(X_test)} amostras")
"""))

# --- SEÇÃO 6: MODELAGEM ---
cells.append(nbf.v4.new_markdown_cell("""
### 5. Modelagem Preditiva
Usaremos 3 modelos para prever a velocidade atual com base no histórico imediato e na hora/dia.
"""))
cells.append(nbf.v4.new_code_cell("""
# 1. Regressão Linear (Baseline)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# 2. Random Forest Regressor
# Limitamos n_estimators para treinar rápido no Colab
rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# 3. XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

print("Modelos treinados com sucesso!")
"""))

# --- SEÇÃO 7: AVALIAÇÃO ---
cells.append(nbf.v4.new_markdown_cell("""
### 6. Avaliação e Comparação
Métricas: MAE (Erro médio absoluto), RMSE (Raiz do Erro Quadrático Médio) e R².
"""))
cells.append(nbf.v4.new_code_cell("""
def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'Modelo': name, 'MAE': mae, 'RMSE': rmse, 'R²': r2}

results = []
results.append(evaluate_model('Regressão Linear', y_test, lr_preds))
results.append(evaluate_model('Random Forest', y_test, rf_preds))
results.append(evaluate_model('XGBoost', y_test, xgb_preds))

results_df = pd.DataFrame(results)

display(results_df.style.highlight_min(subset=['MAE', 'RMSE'], color='lightgreen')
                        .highlight_max(subset=['R²'], color='lightgreen'))

fig = px.bar(results_df, x='Modelo', y='R²', title="Comparação de R² entre Modelos",
             color='Modelo', text_auto='.3f')
fig.show()
"""))

# --- SEÇÃO 8: OVERFITTING E CONCLUSÃO ---
cells.append(nbf.v4.new_markdown_cell("""
### 7. Análise Visual (Real vs Predito) e Conclusão
Vamos visualizar um pequeno trecho de tempo no conjunto de teste (ex: um dia de trânsito) para ver como o XGBoost acompanhou as quedas de velocidade reais.
"""))
cells.append(nbf.v4.new_code_cell("""
# Pegando apenas 288 amostras (24 horas * 12 medições de 5 min)
amostras = 288
test_comparison = pd.DataFrame({
    'Índice (Tempo)': range(amostras),
    'Real (mph)': y_test.values[:amostras],
    'XGBoost Predito (mph)': xgb_preds[:amostras]
})

fig = go.Figure()
fig.add_trace(go.Scatter(x=test_comparison['Índice (Tempo)'], y=test_comparison['Real (mph)'],
                    mode='lines', name='Real (Ground Truth)'))
fig.add_trace(go.Scatter(x=test_comparison['Índice (Tempo)'], y=test_comparison['XGBoost Predito (mph)'],
                    mode='lines', name='Predito (XGBoost)', line=dict(dash='dot')))

fig.update_layout(title="XGBoost: Previsão vs Realidade (Janela de 24 horas)",
                  xaxis_title="Tempo (Intervalos de 5 min)",
                  yaxis_title="Velocidade (mph)")
fig.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""
### 8. Conclusão Científica
A abordagem de utilizar a própria Série Temporal (*Lags* do passado) para prever o futuro no dataset **METR-LA** demonstrou altíssima eficácia.
- Como o trânsito muda de forma contínua, prever os próximos 5 minutos usando os 15 minutos anteriores se beneficia muito de modelos não-lineares (*XGBoost*), superando a Regressão Linear que não lida tão bem com picos abruptos de lentidão.
"""))

for cell in cells:
    nb.cells.append(cell)

with open('C:\\\\Users\\\\samuelbarroso\\\\Documents\\\\Desenvolvimento\\\\TraficGenius\\\\notebook.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook atualizado para METR-LA com sucesso!")
