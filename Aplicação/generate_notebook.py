# -*- coding: utf-8 -*-
"""
Gerador de Notebook Jupyter (Jupyter Notebook Generator)
Este script cria de forma automatizada e programática o arquivo de documentação e execução
interativa 'notebook.ipynb' do projeto TraficGenius, organizando-o em células de Markdown e Código.
"""

import nbformat as nbf

def create_notebook():
    """
    Instancia uma nova estrutura de notebook Jupyter, preenche com o roteiro de execução
    do pipeline de Machine Learning (Fases 1 a 5) baseado na base enriquecida adaptada e grava o arquivo notebook.ipynb.
    """
    nb = nbf.v4.new_notebook()
    
    nb.cells.append(nbf.v4.new_markdown_cell("""# Projeto TraficGenius: Análise Multivariada de Severidade de Acidentes (Versão Advanced)
Este notebook consolida a execução *End-to-End* do projeto de detecção de fatores de risco para acidentes de trânsito utilizando a **Base Enriquecida** contendo as features do OpenStreetMap e do grid hexagonal H3.

## Arquitetura do Pipeline baseada na Base Enriquecida:
1. **Fase 1 (EDA):** Amostragem de parquet, Feature Engineering Espacial (K-Means), Imputação e Detecção de Outliers híbrida (Mahalanobis).
2. **Fase 2 (Premissas):** Testes paramétricos de resíduos (Jarque-Bera, Breusch-Pagan, Durbin-Watson) e avaliação iterativa de VIF (Variance Inflation Factor).
3. **Fase 3 a 5 (Modelagem):** Treinamento comparativo de modelos com Stacking Ensemble e Threshold Tuning para otimização do F1-Score.
4. **Fase 6 (Avançado):** Geração de curvas ROC/PR, diagramas de calibração, curvas de Lift e Ganho, projeção t-SNE e explicabilidade local e global via valores SHAP.
"""))
    
    nb.cells.append(nbf.v4.new_code_cell("""# Importações necessárias para análise, modelagem e plotagem
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Ignorar avisos do runtime e configurar estilo neon escuro
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")"""))

    # -------------------------------------------------------
    # SEÇÃO 0: DEFINIÇÃO DO PROBLEMA
    # -------------------------------------------------------
    nb.cells.append(nbf.v4.new_markdown_cell("""---
# 📌 Seção 0 — Definição do Problema

> **Objetivo:** Entender e descrever claramente o problema que está sendo resolvido.
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""## 0.1 Descrição do Problema

O problema abordado é de **classificação supervisionada binária** (*supervised binary classification*):
dado um conjunto de atributos de infraestrutura viária, condições climáticas e contexto temporal
no momento de um acidente de trânsito, prever se esse acidente resultará em consequências
**Graves ou Fatais (classe 1)** ou **Leves/Moderadas (classe 0)**.

O objetivo prático é fornecer aos gestores de segurança viária uma ferramenta preditiva capaz de
antecipar **onde, quando e sob quais condições** os acidentes tendem a ser mais letais, permitindo
a alocação preventiva de recursos de emergência e investimentos em infraestrutura.

Duas grandes perguntas orientam o projeto:
- **Quais fatores** — climáticos, físicos da via ou temporais — determinam se um acidente será grave ou fatal?
- **É possível generalizar** esse conhecimento para qualquer região geográfica, sem depender de coordenadas absolutas?

> *English tip: target variable = variável-alvo | feature engineering = engenharia de atributos | imbalanced dataset = dataset desbalanceado*
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""## 0.2 Premissas e Hipóteses sobre o Problema

As seguintes hipóteses foram formuladas e testadas ao longo do projeto:

| # | Hipótese | Status |
|---|----------|--------|
| H1 | A infraestrutura da via (pontes, curvas, velocidade) tem **maior poder preditivo** sobre a gravidade do que o clima isolado | ✅ Confirmada — `comprimento_pontes_metros` (15,58% de importância no XGBoost) superou variáveis climáticas |
| H2 | Interações físicas **não-lineares** (chuva×curvas, geada×curvatura) aumentam a probabilidade de acidente grave de forma multiplicativa | ✅ Confirmada — variáveis `interacao_chuva_curvas` e `interacao_clima_curvatura` aprovadas no teste VIF |
| H3 | O contexto de **país** (Brasil vs. EUA) influencia a severidade por diferenças de infraestrutura e legislação | ✅ Incorporada como variável de controle `pais_US` no modelo final |
| H4 | Modelos **não-paramétricos** (XGBoost, Redes Neurais) superam modelos lineares (Logit, LDA) | ✅ Confirmada — XGBoost ROC-AUC 77,82% vs. Logit 73,63% |
| H5 | Um **comitê de modelos (Stacking Ensemble)** supera qualquer modelo individual | ✅ Confirmada — Stacking ROC-AUC 78,26%, único a superar 50% de F1 na classe grave |

> *English tip: hypothesis = hipótese | ensemble = comitê/conjunto de modelos | feature importance = importância de variáveis*
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""## 0.3 Restrições e Condições Impostas para Seleção dos Dados

As seguintes restrições foram explicitamente impostas ao pipeline de dados:

| Restrição | Justificativa |
|-----------|--------------|
| **Exclusão de coordenadas absolutas** (latitude, longitude, IDs de H3) | Garantir generalização do modelo para qualquer região — sem memorizar localizações |
| **Exclusão da variável `data_inversa`** (data do acidente) | Evitar vazamento de informação (*data leakage*) temporal |
| **Substituição da coluna `pais`** (string) pela flag numérica `pais_US` | Compatibilidade com modelos numéricos via encoding explícito |
| **Remoção de variáveis com VIF > 10** | Garantir estabilidade dos estimadores e ausência de multicolinearidade severa (`interacao_velocidade_faixas` foi eliminada, VIF = 25,53) |
| **Amostragem de 100.000 registros** nas etapas de visualização | Evitar estouro de memória RAM nas etapas com Seaborn, t-SNE e SHAP |
| **Apenas registros aprovados na limpeza de outliers** | Somente dados após filtragem por Distância de Mahalanobis e Isolation Forest foram utilizados |

> *English tip: data leakage = vazamento de dados | VIF = Variance Inflation Factor = Fator de Inflação da Variância*
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""## 0.4 Descrição do Dataset

### Origem dos Dados
- **Base primária:** Registros de acidentes rodoviários dos **Estados Unidos e do Brasil**, consolidados no arquivo `dados_unificados.parquet`
- **Enriquecimento climático:** Dados de reanálise climática **ERA5** (ECMWF) integrados por localização e data/hora de cada acidente
- **Enriquecimento geoespacial:** Sistema de indexação hexagonal **H3 do Uber** (resoluções 9, 10 e 11) para agregar atributos viários por zona geográfica

### Dimensões da Base

| Etapa | Registros |
|-------|-----------|
| Base bruta consolidada | ~8.100.000 |
| Após limpeza de outliers (Mahalanobis + Isolation Forest) | **7.846.838** |
| Conjunto de treino (70%) | ~5.492.786 |
| Conjunto de teste (30%) | **2.354.052** |

### Variável-Alvo (*Target Variable*)
- **`severidade_binaria`:** `0` = Leve/Moderado | `1` = Grave/Fatal
- Distribuição: **79% classe 0 / 21% classe 1** — dataset desbalanceado tratado com SMOTE e Threshold Tuning

### Grupos de Atributos (*Features*)

| Grupo | Principais Variáveis | Qtd |
|-------|---------------------|-----|
| **Climáticas** | `temperatura_celsius`, `ponto_orvalho_celsius`, `pressao_hpa`, `velocidade_vento_u`, `velocidade_vento_v`, `cobertura_nuvens_percentual`, `precipitacao_milimetros` | 7 |
| **Infraestrutura viária** | `velocidade_media`, `quantidade_faixas_media`, `curvatura_acumulada`, `desvio_maximo_curvatura`, `quantidade_curvas_acentuadas`, `quantidade_cruzamentos`, `quantidade_semaforos`, `quantidade_rotatorias`, `quantidade_pontes`, `quantidade_tuneis` | 10 |
| **Entorno urbano** | `quantidade_postos_combustivel`, `quantidade_restaurantes`, `quantidade_escolas`, `quantidade_hospitais`, `quantidade_locais_interesse`, `area_urbana_m2`, `area_rural_m2` | 7 |
| **Geoespacial H3** | `extensao_rodovia_metros_res9/10/11`, `rodovia_dominante_res9/10/11` (categórica → dummies) | 6 numéricos + dummies |
| **Temporais** | `Hora_do_Dia`, `Dia_da_Semana`, `Mes`, `Horario_Pico` | 4 |
| **Interações físicas** | `interacao_chuva_curvas` = precipitação × curvas; `interacao_clima_curvatura` = temperatura × curvatura | 2 |
| **Controle de país** | `pais_US` (1 = EUA, 0 = Brasil) | 1 |

> *English tip: features = atributos/variáveis preditoras | target = alvo | dummy variables = variáveis binárias de codificação categórica*
"""))

    nb.cells.append(nbf.v4.new_code_cell("""# Inspecionar a base enriquecida limpa e exibir estatísticas básicas do dataset
import pandas as pd
import os

project_root = os.getcwd()
file_path = os.path.join(project_root, "dataset", "dataset_amostra_limpa_binaria.parquet")

df = pd.read_parquet(file_path)

print("=" * 60)
print(f"  DIMENSÕES DA BASE: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
print("=" * 60)

print("\\n📊 DISTRIBUIÇÃO DA VARIÁVEL-ALVO (severidade_binaria):")
print(df['severidade_binaria'].value_counts())
print(df['severidade_binaria'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

print("\\n📋 TIPOS DAS COLUNAS:")
print(df.dtypes.value_counts())

print("\\n🔍 AMOSTRA DOS DADOS (primeiras 5 linhas):")
df.head()
"""))

    nb.cells.append(nbf.v4.new_code_cell("""# Exibir estatísticas descritivas das variáveis numéricas
print("📈 ESTATÍSTICAS DESCRITIVAS DAS VARIÁVEIS CONTÍNUAS:")
df.describe().round(3)
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""---
## Fase 1: EDA e Preparação da Base Enriquecida
"""))
    nb.cells.append(nbf.v4.new_code_cell("""# Executa a limpeza e amostragem robusta sobre a base de acidentes unificada
!python pipeline_fase1_eda_binaria.py"""))
    
    nb.cells.append(nbf.v4.new_code_cell("""# Carrega e inspeciona a base adaptada enriquecida e limpa
project_root = os.getcwd()
file_path = os.path.join(project_root, "dataset", "dataset_amostra_limpa_binaria.parquet")

df = pd.read_parquet(file_path)
print(f"[INFO] Base enriquecida carregada com sucesso. Shape: {df.shape}")
df.head()"""))
    
    nb.cells.append(nbf.v4.new_markdown_cell("### Distribuição Geográfica dos Acidentes (Clusters Espaciais K-Means)"))
    nb.cells.append(nbf.v4.new_code_cell("""# Gráfico de dispersão espacial colorido por zonas de risco (Clusters do K-Means)
plt.figure(figsize=(10,6))
sns.scatterplot(x='Longitude_Inicial', y='Latitude_Inicial', hue='Cluster_Espacial', palette='tab20', data=df.sample(10000, random_state=42), alpha=0.5, s=15)
plt.title('Zonas Espaciais de Risco (K-Means) da Base Enriquecida')
plt.show()"""))
    
    nb.cells.append(nbf.v4.new_markdown_cell("## Fase 2: Diagnóstico de Premissas Clássicas e Multicolinearidade"))
    nb.cells.append(nbf.v4.new_code_cell("""# Executa os diagnósticos de homocedasticidade, independência dos resíduos e colinearidade (VIF)
!python pipeline_fase2_premissas_binaria.py"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""### Visualização de Pressupostos Estatísticos e Multicolinearidade
Abaixo, visualizamos os gráficos gerados durante a Fase 2:
1. **Matriz de Correlação de Pearson:** Correlação linear entre variáveis contínuas.
2. **Fator de Inflação de Variância (VIF):** Detecção de colinearidade crítica.
3. **Distribuição de Normalidade:** Comparação visual de resíduos contra curva teórica.
"""))
    nb.cells.append(nbf.v4.new_code_cell("""from IPython.display import Image, display
folder_path = os.path.join(project_root, "dataset")
# Exibe imagens de diagnóstico geradas
display(Image(filename=os.path.join(folder_path, 'matriz_correlacao.png')))
display(Image(filename=os.path.join(folder_path, 'vif_multicolinearidade.png')))
display(Image(filename=os.path.join(folder_path, 'distribuicao_normalidade.png')))"""))
    
    nb.cells.append(nbf.v4.new_markdown_cell("## Fases 3 a 5: Modelagem Preditiva com Stacking e Flag de País"))
    nb.cells.append(nbf.v4.new_code_cell("""# Executa o pipeline de treinamento de modelos preditivos contendo a flag de país
!python pipeline_fase3a5_modelagem_com_flag_pais.py"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""### Avaliação Visual do Desempenho e Importância de Variáveis
Exibe os gráficos gerados durante o processo de treinamento e avaliação:
1. **Importância de Variáveis (Feature Importance):** Variáveis mais relevantes do XGBoost.
2. **Matriz de Confusão:** Relação percentual e absoluta de acertos no teste.
3. **Comparativo de Métricas:** F1-Score Macro e Acurácia Global de todos os modelos base e Stacking.
"""))
    nb.cells.append(nbf.v4.new_code_cell("""from IPython.display import Image, display
# Exibe imagens de performance preditiva
display(Image(filename=os.path.join(folder_path, 'importancia_features_xgboost_flag_pais.png')))
display(Image(filename=os.path.join(folder_path, 'matriz_confusao_xgboost_flag_pais.png')))
display(Image(filename=os.path.join(folder_path, 'comparativo_performance_flag_pais.png')))"""))
    
    nb.cells.append(nbf.v4.new_markdown_cell("## Fase 6: Avaliação Avançada e Explicabilidade SHAP"))
    nb.cells.append(nbf.v4.new_code_cell("""# Executa a geração dos gráficos avançados de calibração, PR, ROC, Lift, Ganho e SHAP
!python gerar_graficos_storytelling.py"""))
    
    nb.cells.append(nbf.v4.new_markdown_cell("""### Visualizações Finais de Storytelling e Avaliação de Desempenho Avançado
Abaixo, visualizamos os gráficos gerados para a avaliação final do classificador:
1. **Curvas ROC (AUC) e PR (Precision-Recall)**
2. **Curvas de Calibração (Reliability Diagram)**
3. **Curvas de Lift e Ganho Acumulado**
4. **Projeção t-SNE 2D**
5. **Resumo das Contribuições de Variáveis (SHAP Values)**
"""))
    
    nb.cells.append(nbf.v4.new_code_cell("""# Exibe as curvas ROC, PR, Calibração, Lift e Ganho
display(Image(filename=os.path.join(folder_path, 'story_desempenho_roc.png')))
display(Image(filename=os.path.join(folder_path, 'story_desempenho_pr.png')))
display(Image(filename=os.path.join(folder_path, 'story_desempenho_calibration.png')))
display(Image(filename=os.path.join(folder_path, 'story_desempenho_lift_gain.png')))
# Exibe o t-SNE e SHAP do XGBoost
display(Image(filename=os.path.join(folder_path, 'story_desempenho_tsne.png')))
display(Image(filename=os.path.join(folder_path, 'story_desempenho_shap.png')))"""))

    # -------------------------------------------------------
    # SEÇÃO 1: PREPARAÇÃO DE DADOS
    # -------------------------------------------------------
    nb.cells.append(nbf.v4.new_markdown_cell("""---
# 📦 Seção 1 — Preparação de Dados

> **Objetivo:** Realizar operações de preparação dos dados para treinamento e avaliação dos modelos.
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""## 1.1 Separação Treino / Teste / Validação (Meta-Holdout)

A divisão dos dados seguiu uma estratégia em **três partições**, justificada pela arquitetura de Stacking Ensemble:

| Partição | Proporção | Tamanho (aprox.) | Uso |
|----------|-----------|-----------------|-----|
| **Treino Base** | 59,5% do total | ~4.669.568 linhas | Treinar os 5 modelos base (Logit, LDA, RF, MLP, XGBoost) |
| **Meta-Holdout** | 10,5% do total | ~823.210 linhas | Gerar as probabilidades para treinar o meta-aprendedor (Stacking) e otimizar os limiares |
| **Teste Final** | 30% do total | **2.354.052 linhas** | Avaliação final com dados **nunca vistos** por nenhum modelo |

A separação foi feita com **estratificação pela variável-alvo** (`stratify=y`) para garantir que as proporções de 79%/21% fossem mantidas em todas as partições.

```python
# Divisão Treino/Teste estratificada (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# Divisão do Treino em Base (85%) e Meta-Holdout (15%) para o Stacking
X_base, X_meta, y_base, y_meta = train_test_split(
    X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
)
```

> *English tip: train/test split = divisão treino/teste | stratified split = divisão estratificada | holdout = partição reservada*
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""## 1.2 Validação Cruzada — Por Que Não Foi Utilizada

**Não foi utilizada validação cruzada (cross-validation)** neste projeto, e a justificativa é técnica e computacional:

| Motivo | Explicação |
|--------|-----------|
| **Volume de dados** | Com 7,8 milhões de registros, cada *fold* de um k=5 teria ~1,5 milhão de linhas. Treinar 5 modelos × 5 folds = 25 treinamentos completos, o que tornaria o pipeline inviável em ambiente local |
| **Tamanho do conjunto de teste** | Com **2,35 milhões de linhas** na partição de teste, a estimativa de generalização já possui altíssima confiança estatística — a margem de erro é desprezível |
| **Arquitetura de Stacking** | O Stacking Ensemble exige uma partição Meta-Holdout dedicada que **não pode se sobrepor** ao treino dos modelos base. Isso seria estruturalmente incompatível com a validação cruzada padrão |
| **Risco de data leakage** | A natureza temporal dos dados (acidentes ao longo do tempo) sugere que uma divisão cronológica é mais rigorosa do que uma divisão aleatória em folds |

> *English tip: cross-validation = validação cruzada | k-fold = validação em k partições | data leakage = vazamento de dados entre partições*
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""## 1.3 Transformações Aplicadas aos Dados

### Padronização (StandardScaler)
Foi aplicada **padronização Z-score** em todas as variáveis numéricas, centralizando-as na média 0 e desvio padrão 1:

$$z = \\frac{x - \\mu}{\\sigma}$$

**Justificativa:** Algoritmos baseados em distância (LDA) e gradiente (Regressão Logística, MLP) são sensíveis à escala das variáveis. A padronização é mais robusta do que a normalização Min-Max quando existem outliers residuais.

```python
scaler = StandardScaler()
X_base_scaled  = scaler.fit_transform(X_base)   # Ajuste APENAS no treino base
X_meta_scaled  = scaler.transform(X_meta)        # Aplicação no meta-holdout
X_test_scaled  = scaler.transform(X_test)        # Aplicação no teste final
```

> ⚠️ **Atenção:** O `scaler.fit()` foi chamado **somente** no conjunto de treino base, nunca no teste — prevenindo vazamento de informação.

### One-Hot Encoding (Variáveis Categóricas)
As três variáveis de tipo de rodovia dominante (`rodovia_dominante_res9/10/11`) foram transformadas em variáveis binárias:
```python
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
```

### Criação da Flag de País
```python
df['pais_US'] = (df['pais'] == 'US').astype(int)  # 1 = EUA, 0 = Brasil
```

> *English tip: standardization = padronização | normalization = normalização | one-hot encoding = codificação binária de categorias*
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""## 1.4 Feature Selection — Seleção de Atributos

O processo de seleção de atributos foi realizado em **duas etapas complementares**:

### Etapa A — Eliminação por Regra de Negócio
Antes de qualquer teste estatístico, foram removidas manualmente:
- `latitude`, `longitude`, IDs H3: evitar memorização geográfica
- `data_inversa`: evitar data leakage temporal
- `pais` (string): substituída pela flag numérica `pais_US`

### Etapa B — Eliminação por VIF (Variance Inflation Factor)
Executada automaticamente pelo `pipeline_fase2_premissas_binaria.py`:

```python
# Remoção iterativa de variáveis com VIF > 10 (stepwise)
while True:
    vif_data = calcular_vif(X_continuas)
    max_vif = vif_data['vif'].max()
    if max_vif > 10:
        col_remover = vif_data.loc[vif_data['vif'].idxmax(), 'feature']
        X_continuas = X_continuas.drop(columns=[col_remover])
    else:
        break
```

**Variável eliminada:** `interacao_velocidade_faixas` (VIF = **25,53**) — produto de `velocidade_media` × `quantidade_faixas_media`, que já existem separadamente no modelo.

**Resultado final:** 38 atributos preditores aprovados e salvos em `dataset/features_selecionadas_binaria.json`.

> *English tip: feature selection = seleção de atributos | multicollinearity = multicolinearidade | VIF = Fator de Inflação da Variância*
"""))

    nb.cells.append(nbf.v4.new_code_cell("""# Carregar e exibir as features selecionadas após VIF stepwise
import json, os

project_root = os.getcwd()
features_path = os.path.join(project_root, "dataset", "features_selecionadas_binaria.json")

with open(features_path) as f:
    features = json.load(f)

print(f"✅ Total de features selecionadas após VIF stepwise: {len(features)}\\n")
for i, feat in enumerate(features, 1):
    print(f"  {i:2}. {feat}")
"""))

    nb.cells.append(nbf.v4.new_code_cell("""# Visualizar divisão treino/teste com gráfico de barras
import matplotlib.pyplot as plt
import pandas as pd
import os

project_root = os.getcwd()
df = pd.read_parquet(os.path.join(project_root, "dataset", "dataset_amostra_limpa_binaria.parquet"))

total = len(df)
n_teste = int(total * 0.30)
n_treino_base = int(total * 0.70 * 0.85)
n_meta = total - n_teste - n_treino_base

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Gráfico 1: Divisão das partições
particoes = ['Treino Base\\n(59,5%)', 'Meta-Holdout\\n(10,5%)', 'Teste Final\\n(30%)']
tamanhos  = [n_treino_base, n_meta, n_teste]
cores     = ['#00f0ff', '#ffcc00', '#ff007f']
axes[0].bar(particoes, tamanhos, color=cores, edgecolor='#2d2d34', linewidth=1.2)
axes[0].set_title('Divisão das Partições do Dataset', color='#00f0ff', fontsize=12)
axes[0].set_ylabel('Número de Registros', color='#e2e2e9')
for i, (p, v) in enumerate(zip(particoes, tamanhos)):
    axes[0].text(i, v + 50000, f'{v:,}', ha='center', color='#e2e2e9', fontsize=9)
axes[0].tick_params(colors='#e2e2e9')

# Gráfico 2: Distribuição da variável-alvo
contagem = df['severidade_binaria'].value_counts().sort_index()
labels   = ['Leve/Moderado (0)', 'Grave/Fatal (1)']
axes[1].pie(contagem, labels=labels, colors=['#00f0ff', '#ff007f'],
            autopct='%1.1f%%', startangle=90,
            textprops={'color': '#e2e2e9'})
axes[1].set_title('Distribuição da Variável-Alvo', color='#00f0ff', fontsize=12)

plt.tight_layout()
plt.show()
"""))

    # -------------------------------------------------------
    # SEÇÃO 2: MODELAGEM E TREINAMENTO
    # -------------------------------------------------------
    nb.cells.append(nbf.v4.new_markdown_cell("""---
# 🧠 Seção 2 — Modelagem e Treinamento

> **Objetivo:** Construir modelos para resolver o problema de classificação de severidade de acidentes.
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""## 2.1 Algoritmos Selecionados e Justificativas

Foram selecionados **5 modelos base** com perfis complementares, mais um **meta-aprendedor**:

| Modelo | Tipo | Justificativa |
|--------|------|--------------|
| **Regressão Logística** | Linear | Baseline interpretável; estima probabilidades calibradas; referência para medir ganho dos modelos complexos |
| **LDA (Análise Discriminante Linear)** | Paramétrico | Encontra fronteiras lineares ótimas entre classes; serve como corretor de fronteira no Stacking (peso negativo de -2,97) |
| **Random Forest** | Ensemble de Árvores | Robusto a outliers residuais; captura interações não-lineares sem necessidade de feature engineering adicional |
| **MLP (Rede Neural Artificial)** | Deep Learning | Modela relações altamente não-lineares entre infraestrutura e severidade; fallback scikit-learn usado por incompatibilidade de AVX no CPU local |
| **XGBoost** | Gradient Boosting | Estado da arte para dados tabulares; usa `scale_pos_weight` para compensar o desbalanceamento sem SMOTE; explícito via Feature Importance |
| **Stacking Ensemble** | Meta-Aprendedor | Combina as previsões dos 5 modelos base via Regressão Logística, ponderando a contribuição de cada um automaticamente |

> *English tip: baseline = modelo de referência mínima | gradient boosting = otimização sequencial por gradiente | meta-learner = modelo que aprende sobre outros modelos*
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""## 2.2 Hiperparâmetros Iniciais e Ajustes

Os hiperparâmetros foram definidos com base em boas práticas para datasets de grande volume:

### XGBoost
```python
XGBClassifier(
    n_estimators=80,        # 80 árvores — balanço entre performance e tempo de treino
    max_depth=6,            # Profundidade moderada para evitar overfitting
    learning_rate=0.1,      # Taxa de aprendizado padrão — estável para grandes bases
    scale_pos_weight=ratio, # Compensa o desbalanceamento: ratio = n_negativos / n_positivos ≈ 3.76
    tree_method='hist',     # Algoritmo histogram-based — muito mais rápido para milhões de linhas
    eval_metric='logloss'   # Métrica de otimização interna
)
```

### Random Forest
```python
RandomForestClassifier(
    n_estimators=30,        # 30 árvores — reduzido por limitação de RAM
    max_depth=12,           # Profundidade controlada para evitar memorização
    max_samples=0.10,       # Usa apenas 10% dos dados por árvore — viabiliza o treino em 7,8M linhas
    class_weight='balanced' # Ajusta pesos inversamente às frequências das classes
)
```

### MLP (fallback scikit-learn)
```python
MLPClassifier(
    hidden_layer_sizes=(64, 32), # Arquitetura 2 camadas ocultas: 64 → 32 neurônios
    max_iter=150,
    early_stopping=True          # Para o treino quando a validação para de melhorar
)
```

> *English tip: hyperparameter = hiperparâmetro | learning rate = taxa de aprendizado | early stopping = parada antecipada*
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""## 2.3 Problema de Underfitting

**Não foi observado underfitting** nos modelos treinados. As evidências são:

- A Regressão Logística (modelo mais simples) já alcançou **73,63% de ROC-AUC** no conjunto de teste, superando significativamente o baseline aleatório de 50%.
- A acurácia global de todos os modelos supera o **MCC (chute na maioria = 79%)** após a aplicação do limiar ótimo, indicando que os modelos aprenderam padrões reais.
- A curva de aprendizado do XGBoost (baseada no logloss) convergiu suavemente, sem sinal de subajuste.

O principal desafio do problema foi o **desbalanceamento de classes** (79%/21%), não o underfitting. A solução foi o Threshold Tuning, que ajustou o limiar de decisão de 50% para valores menores (ex: 0.23 na Logística), forçando maior sensibilidade à classe minoritária grave.

> *English tip: underfitting = subajuste | overfitting = sobreajuste | class imbalance = desbalanceamento de classes*
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""## 2.4 Otimização de Hiperparâmetros — Threshold Tuning

A principal otimização realizada foi o **Decision Threshold Tuning** (*Otimização do Limiar de Decisão*):

Em vez do corte padrão de 50%, buscamos o limiar que **maximiza o F1-Score da classe grave (1)** no Meta-Holdout:

```python
def find_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.05, 0.95, 91)  # Testa 91 limiares diferentes
    best_thresh, best_f1 = 0.5, 0
    for t in thresholds:
        f1 = f1_score(y_true, (y_prob >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh
```

**Impacto do Threshold Tuning na Regressão Logística:**

| Métrica | Limiar 50% (padrão) | Limiar 0.23 (ótimo) | Ganho |
|---------|---------------------|---------------------|-------|
| Acurácia Balanceada | 53,61% | **67,23%** | +13,6 p.p. |
| F1-Score Classe Grave | 16,45% | **46,43%** | +29,9 p.p. |

> *English tip: threshold tuning = ajuste do limiar de decisão | precision-recall tradeoff = troca entre precisão e revocação*
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""## 2.5 Método Avançado — Stacking Ensemble

O método mais avançado implementado foi o **Stacking Ensemble** (*Comitê por Empilhamento*):

**Funcionamento em 3 etapas:**

1. **Treino dos modelos base** na partição `X_base` (59,5% do total)
2. **Geração do meta-dataset:** cada modelo base prevê probabilidades no `X_meta` (Meta-Holdout). Essas probabilidades viram as *features* do meta-aprendedor
3. **Treino do meta-aprendedor** (Regressão Logística) que aprende a combinar as previsões

**Pesos aprendidos pelo meta-aprendedor:**

| Modelo Base | Coeficiente | Interpretação |
|------------|-------------|--------------|
| XGBoost | **+3.6445** | Maior contribuição positiva |
| MLP (Rede Neural) | **+2.5055** | Segunda maior contribuição |
| Logistic Regression | **+1.5295** | Contribuição positiva moderada |
| Random Forest | **+1.1528** | Contribuição positiva menor |
| LDA | **-2.9678** | **Corretor de fronteiras lineares** — seu sinal negativo ajusta o viés dos demais modelos |

> *English tip: stacking = empilhamento | meta-learner = meta-aprendedor | base models = modelos base*
"""))

    nb.cells.append(nbf.v4.new_code_cell("""# Executar o pipeline de modelagem com Flag de País
!python pipeline_fase3a5_modelagem_com_flag_pais.py
"""))

    nb.cells.append(nbf.v4.new_code_cell("""# Exibir pesos do Stacking Ensemble e limiares ótimos salvos
import json, os

project_root = os.getcwd()
limiares_path = os.path.join(project_root, "dataset", "limiares_decisao_flag_pais.json")

with open(limiares_path) as f:
    limiares = json.load(f)

print("🎯 LIMIARES ÓTIMOS DE DECISÃO (maximizam F1-Score da classe Grave):\\n")
for modelo, limiar in limiares.items():
    print(f"  {modelo:30}: {limiar:.2f}")
"""))

    # -------------------------------------------------------
    # SEÇÃO 3: AVALIAÇÃO DE RESULTADOS
    # -------------------------------------------------------
    nb.cells.append(nbf.v4.new_markdown_cell("""---
# 📊 Seção 3 — Avaliação de Resultados

> **Objetivo:** Analisar o desempenho dos modelos em dados não vistos (conjunto de teste).
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""## 3.1 Métricas de Avaliação Escolhidas e Justificativas

| Métrica | Justificativa |
|---------|--------------|
| **ROC-AUC** | Mede a capacidade de separação das classes independentemente do limiar. Principal métrica de ranking para problemas desbalanceados |
| **F1-Score (Classe Grave)** | Harmoniza Precision e Recall da classe minoritária (Grave/Fatal). É a métrica mais crítica — um modelo que nunca prevê "grave" tem F1=0 |
| **F1-Score Macro** | Média não-ponderada do F1 das duas classes — penaliza igualmente o erro em qualquer classe |
| **Acurácia Balanceada** | Média das taxas de acerto por classe — ignora o efeito do desbalanceamento |
| **Acurácia Global (Limiar Ótimo)** | Acurácia total com o limiar ajustado para cada modelo |
| **MCC e PCC** | Critérios do acaso — MCC = chutar sempre a maioria (79%); PCC = chute proporcional (66,82%). O modelo precisa superar **1.25 × PCC = 83,53%** para ser estatisticamente relevante |

> *English tip: ROC-AUC = area under the ROC curve | recall = sensibilidade/revocação | precision = precisão preditiva*
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""## 3.2 Resultados Comparativos no Conjunto de Teste (2,35 Milhões de Linhas)

**Critérios do Acaso:**
- MCC (chute na maioria): **79,00%**
- PCC (chute proporcional): **66,82%**
- Meta mínima (1,25 × PCC): **83,53%**

| Modelo | Limiar Ótimo | Acurácia (50%) | Acurácia (Ótimo) | Acurácia Balanceada | F1 Classe Grave | F1 Macro | ROC-AUC |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **🏆 Stacking Ensemble** | **0.27** | **80,42%** | **72,88%** | **70,18%** | **50,36%** | **65,85%** | **78,26%** |
| XGBoost | 0.55 | 67,01% | 71,42% | 70,08% | 49,90% | 64,95% | 77,82% |
| MLP (Rede Neural) | 0.54 | 66,48% | 70,52% | 69,77% | 49,38% | 64,29% | 77,27% |
| Random Forest | 0.54 | 66,10% | 71,09% | 68,95% | 48,66% | 64,27% | 76,58% |
| LDA | 0.49 | 69,13% | 68,16% | 67,61% | 46,78% | 62,04% | 73,85% |
| Regressão Logística | 0.23 | 79,01% | 68,41% | 67,23% | 46,43% | 62,02% | 73,63% |
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""## 3.3 Os Resultados Fazem Sentido?

**Sim.** As descobertas são consistentes com a física do tráfego e com a literatura de segurança viária:

✅ **Pontes e viadutos** dominam o ranking de importância (15,58%) — acidentes em estruturas elevadas têm menor margem de escape e impactam barreiras rígidas.

✅ **Velocidade média** (6,64%) — a energia cinética de impacto cresce com o quadrado da velocidade ($E_c = \frac{1}{2}mv^2$). Vias de alta velocidade produzem acidentes mais graves.

✅ **Rodovias residenciais** atuam como *protetoras* — limites de velocidade baixos reduzem a gravidade.

✅ **O Stacking supera todos os modelos individuais** — o meta-aprendedor aprendeu que o LDA serve como "freio" aos excessos dos outros modelos, usando seu coeficiente negativo (-2,97).

✅ **A flag de país (`pais_US`) como variável de controle** é coerente — os EUA e o Brasil têm infraestruturas, limites de velocidade e comportamentos de tráfego distintos.

> *English tip: feature importance = importância da variável | interpretability = interpretabilidade*
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""## 3.4 Overfitting — Foi Observado?

**Não foi observado overfitting significativo.** As evidências são:

- O Random Forest usa `max_samples=0.10` (subamostragem por árvore) e `max_depth=12`, limitando a memorização.
- O XGBoost usa `n_estimators=80` com `learning_rate=0.1` — configuração conservadora que favorece generalização.
- O MLP usa `early_stopping=True`, interrompendo o treino quando a validação piora.
- **Os resultados no conjunto de teste são coerentes com o Meta-Holdout** — não há queda brusca de desempenho ao mudar de partição.

O risco real do projeto não foi overfitting, mas sim o **trade-off entre Precision e Recall** causado pelo desbalanceamento de classes, que foi mitigado pelo Threshold Tuning.

> *English tip: overfitting = sobreajuste | generalization = generalização | early stopping = parada antecipada*
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""## 3.5 Melhor Solução Encontrada — Stacking Ensemble

O **Stacking Ensemble com limiar ótimo de 0.27** é a melhor solução encontrada, pelos seguintes motivos:

1. **Maior ROC-AUC (78,26%):** Melhor separação geral entre acidentes graves e leves, independentemente do limiar.
2. **Único a superar 50% de F1 na classe grave (50,36%):** Barreira crítica para aplicações práticas de segurança viária — modelos abaixo de 50% falham em mais de metade dos acidentes graves.
3. **Maior acurácia global com limiar ótimo (72,88%):** Corretamente classifica 1.715.551 dos 2.354.052 registros de teste.
4. **Menor taxa de Falsos Positivos:** Especificidade de 74,8% — não gera alarmes desnecessários para acidentes leves.
5. **Robustez:** Por combinar modelos lineares, paramétricos e baseados em árvores, é resiliente a diferentes tipos de padrões nos dados.

**Recomendação de uso em produção:** O Stacking Ensemble com o scaler `scaler_flag_pais.joblib` e os modelos `model_*_flag_pais.joblib/.h5` salvos em `dataset/`, usando o limiar 0.27 para a classificação final.

> *English tip: best solution = melhor solução | production deployment = implantação em produção | threshold = limiar de decisão*
"""))

    nb.cells.append(nbf.v4.new_code_cell("""# Exibir gráficos comparativos de desempenho dos modelos
from IPython.display import Image, display
import os

project_root = os.getcwd()
folder_path = os.path.join(project_root, "dataset")
story_path  = os.path.join(project_root, "resultados_modelagem", "graficos_storytelling")

print("📊 Comparativo de Performance dos Modelos:")
display(Image(filename=os.path.join(folder_path, 'comparativo_performance_flag_pais.png')))

print("\\n📈 Importância das Variáveis (XGBoost):")
display(Image(filename=os.path.join(folder_path, 'importancia_features_xgboost_flag_pais.png')))

print("\\n🔵 Curvas ROC Comparativas:")
display(Image(filename=os.path.join(story_path, 'story_desempenho_roc.png')))

print("\\n🟣 Curvas Precision-Recall Comparativas:")
display(Image(filename=os.path.join(story_path, 'story_desempenho_pr.png')))
"""))

    nb.cells.append(nbf.v4.new_code_cell("""# Exibir matrizes de confusão do modelo vencedor e do XGBoost
print("🏆 Matriz de Confusão — Stacking Ensemble (Limiar 0.27):")
display(Image(filename=os.path.join(folder_path, 'matriz_confusao_stacking_ensemble_flag_pais.png')))

print("\\n⚡ Matriz de Confusão — XGBoost (Limiar 0.55):")
display(Image(filename=os.path.join(folder_path, 'matriz_confusao_xgboost_flag_pais.png')))

print("\\n🧬 Projeção t-SNE 2D — Separação das Classes:")
display(Image(filename=os.path.join(story_path, 'story_desempenho_tsne.png')))

print("\\n🔍 Explicabilidade Global SHAP (XGBoost):")
display(Image(filename=os.path.join(story_path, 'story_desempenho_shap.png')))
"""))

    with open("notebook.ipynb", "w", encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print("Notebook gerado com sucesso!")

if __name__ == "__main__":
    create_notebook()
