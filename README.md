# TraficGenius2 — Notebook de Análise e Modelagem Preditiva de Acidentes

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samuelsouzabsb/TraficGenius/blob/main/TraficGenius2.ipynb)

---

## O que é esse notebook?

Esse notebook documenta e executa todo o processo de análise estatística e modelagem preditiva do projeto **TraficGenius**, que tem como objetivo entender quais fatores tornam um acidente de trânsito grave ou fatal — e construir um modelo capaz de prever isso antes do acidente acontecer.

O trabalho vai desde a exploração inicial dos dados até a avaliação final dos modelos treinados, passando por etapas de diagnóstico estatístico, engenharia de variáveis e comparação entre diferentes algoritmos de machine learning.

---

## O problema que estamos resolvendo

A pergunta central é simples: **dado o que sabemos sobre uma via e sobre as condições no momento do acidente, conseguimos prever se ele vai ser grave?**

Mais formalmente, trata-se de um problema de **classificação binária supervisionada**: cada acidente é classificado como *Leve/Moderado (classe 0)* ou *Grave/Fatal (classe 1)*, com base em variáveis de infraestrutura, clima e tempo.

A motivação prática é oferecer aos gestores de segurança viária uma ferramenta para antecipar onde e quando os acidentes tendem a ser mais sérios, permitindo alocação preventiva de recursos.

Duas perguntas guiam o projeto:
- Quais fatores — da via, do clima ou do horário — determinam a gravidade de um acidente?
- É possível generalizar esse conhecimento para qualquer região, sem depender de coordenadas geográficas específicas?

---

## Hipóteses testadas

Antes de construir qualquer modelo, foram formuladas cinco hipóteses sobre o problema. Todas foram verificadas ao longo do notebook:

| # | O que achávamos | O que encontramos |
|---|-----------------|-------------------|
| H1 | A infraestrutura da via importa mais do que o clima | ✅ Confirmado — comprimento de pontes foi a variável mais importante (15,58% no XGBoost) |
| H2 | Chuva em curvas e frio em pistas sinuosas aumentam a gravidade de forma multiplicativa | ✅ Confirmado — as interações foram aprovadas no teste de multicolinearidade (VIF) |
| H3 | O país (Brasil ou EUA) influencia a severidade | ✅ Incorporado como variável de controle no modelo final |
| H4 | Modelos não-lineares (XGBoost, Redes Neurais) superam modelos lineares | ✅ Confirmado — XGBoost atingiu ROC-AUC de 77,82% contra 73,63% da Regressão Logística |
| H5 | Um comitê de modelos (Stacking Ensemble) supera qualquer modelo individual | ✅ Confirmado — o Stacking atingiu 78,26% de ROC-AUC e foi o único a superar 50% de F1 para acidentes graves |

---

## De onde vieram os dados?

A base foi construída a partir de quatro fontes diferentes:

| Fonte | O que contém | Link |
|-------|-------------|------|
| PRF (Brasil) | Registros de acidentes nas rodovias federais brasileiras | [dados.gov.br](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf) |
| US Accidents (Kaggle) | Acidentes nos Estados Unidos entre 2016 e 2023 | [kaggle.com](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) |
| Geofabrik / OpenStreetMap | Dados de infraestrutura viária (tipo de via, curvas, pontes, semáforos...) | [geofabrik.de](https://download.geofabrik.de/) |
| ERA5 / ECMWF | Dados climáticos de reanálise (temperatura, vento, precipitação...) | [copernicus.eu](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) |

Os dados do OpenStreetMap foram agregados usando o sistema hexagonal **H3 do Uber** (resoluções 9, 10 e 11), que divide o mapa em células hexagonais e permite calcular estatísticas de infraestrutura por zona geográfica.

### Dimensões da base

| Etapa | Quantidade de registros |
|-------|------------------------|
| Base bruta consolidada | ~8.100.000 |
| Após limpeza de outliers | **7.846.838** |
| Treino (70%) | ~5.492.786 |
| Teste (30%) | **2.354.052** |

A variável-alvo (`severidade_binaria`) está **desbalanceada**: 79% dos acidentes são leves e 21% são graves. Isso foi tratado com ajuste de limiares de decisão (Threshold Tuning).

---

## Quais variáveis foram usadas?

Ao final do processo de seleção, 38 variáveis preditoras foram mantidas, distribuídas em 7 grupos:

| Grupo | Exemplos de variáveis |
|-------|----------------------|
| Climáticas | temperatura, ponto de orvalho, precipitação, velocidade do vento |
| Infraestrutura viária | velocidade média, número de faixas, curvas, pontes, túneis, semáforos |
| Entorno urbano | hospitais, escolas, postos de combustível, área urbana vs. rural |
| Geoespacial H3 | extensão de rodovias por hexágono, tipo de via dominante |
| Temporais | hora do dia, dia da semana, mês, horário de pico |
| Interações físicas | chuva × curvas, temperatura × curvatura |
| Controle de país | flag binária (1 = EUA, 0 = Brasil) |

Três variáveis foram **excluídas antes da modelagem**:
- Latitude, longitude e IDs dos hexágonos H3 → evitar que o modelo "memorize" localizações
- Data do acidente → evitar vazamento temporal de informação
- A interação velocidade × faixas → foi eliminada pelo teste de multicolinearidade (VIF = 25,53)

---

## Como o notebook está organizado?

### Seção 0 — Definição do Problema
Descreve o problema, as hipóteses e a estrutura do dataset. É a parte introdutória, que contextualiza tudo que vem a seguir.

### Seção 1 — Preparação dos Dados

**Divisão treino/teste:**
O conjunto de dados foi dividido em três partes:
- **59,5%** para treinar os modelos base
- **10,5%** como *meta-holdout* — usado exclusivamente para treinar o modelo de Stacking e otimizar os limiares de decisão
- **30%** para o teste final, com dados que nenhum modelo viu durante o treinamento

**Por que não foi usada validação cruzada?**
Com 7,8 milhões de registros, fazer 5 folds significaria 25 treinamentos completos — inviável em ambiente local. Além disso, o Stacking Ensemble requer uma partição dedicada que não pode se sobrepor ao treino, o que é incompatível com a validação cruzada padrão. Com 2,35 milhões de linhas no teste, a estimativa de generalização já é estatisticamente robusta.

**Transformações aplicadas:**
- **StandardScaler (Z-score):** padronização das variáveis numéricas — necessária para LDA, Regressão Logística e MLP, que são sensíveis à escala
- **One-Hot Encoding:** codificação das variáveis categóricas de tipo de rodovia
- **VIF stepwise:** remoção iterativa de variáveis com multicolinearidade severa (VIF > 10)

> ⚠️ O StandardScaler foi ajustado **apenas no conjunto de treino** e depois aplicado ao teste — isso é fundamental para não "vazar" informação do futuro para o modelo.

### Seção 2 — Modelagem e Treinamento

**Os cinco modelos treinados:**

| Modelo | Por que foi escolhido |
|--------|----------------------|
| Regressão Logística | Baseline interpretável; referência para medir o ganho dos modelos mais complexos |
| LDA | Encontra fronteiras lineares ótimas; no Stacking, atua como "corretor" com peso negativo |
| Random Forest | Robusto a outliers; captura padrões não-lineares sem engenharia adicional |
| MLP (Rede Neural) | Modela relações altamente não-lineares; fallback para scikit-learn (sem TensorFlow/AVX) |
| XGBoost | Estado da arte para dados tabulares; usa `scale_pos_weight` para compensar o desbalanceamento |

**Stacking Ensemble:**
Os cinco modelos base foram combinados por um **meta-aprendedor** (Regressão Logística), que aprendeu o quanto confiar em cada um:

| Modelo base | Peso aprendido | Interpretação |
|-------------|---------------|---------------|
| XGBoost | +3,64 | Maior contribuição positiva |
| MLP | +2,51 | Segunda maior contribuição |
| Logistic Regression | +1,53 | Contribuição positiva moderada |
| Random Forest | +1,15 | Contribuição positiva menor |
| LDA | **−2,97** | Corretor de viés — reduz excessos dos demais |

**Threshold Tuning:**
Em vez do corte padrão de 50%, buscamos o limiar que **maximiza o F1-Score para acidentes graves**. O impacto foi significativo:

| Modelo | F1-Score (limiar 50%) | F1-Score (limiar ótimo) |
|--------|----------------------|------------------------|
| Regressão Logística | 16,45% | **46,43%** (+29,9 p.p.) |
| Stacking Ensemble | — | **50,36%** |

### Seção 3 — Avaliação dos Resultados

**Métricas usadas e por quê:**

- **ROC-AUC:** mede separação das classes independentemente do limiar — ideal para datasets desbalanceados
- **F1-Score (classe grave):** mais importante do que acurácia geral, pois penaliza igualmente falsos positivos e falsos negativos na classe que importa
- **Acurácia Balanceada:** média de acertos por classe, ignorando o efeito do desbalanceamento
- **MCC e PCC:** critérios do acaso — qualquer modelo precisa superar 1,25 × PCC = 83,53% de acurácia para ser estatisticamente relevante

**Resultados comparativos no conjunto de teste (2,35 milhões de linhas):**

| Modelo | Limiar ótimo | Acurácia Global | Acurácia Balanceada | F1 (Grave) | ROC-AUC |
|--------|:---:|:---:|:---:|:---:|:---:|
| 🏆 **Stacking Ensemble** | **0,27** | **72,88%** | **70,18%** | **50,36%** | **78,26%** |
| XGBoost | 0,55 | 71,42% | 70,08% | 49,90% | 77,82% |
| MLP | 0,54 | 70,52% | 69,77% | 49,38% | 77,27% |
| Random Forest | 0,54 | 71,09% | 68,95% | 48,66% | 76,58% |
| LDA | 0,49 | 68,16% | 67,61% | 46,78% | 73,85% |
| Regressão Logística | 0,23 | 68,41% | 67,23% | 46,43% | 73,63% |

**Os resultados fazem sentido?**

Sim. As descobertas são coerentes com o que sabemos sobre física do tráfego:
- Pontes e viadutos dominam a importância (15,58%) — estruturas elevadas têm menor margem de escape e impactos mais rígidos
- Velocidade média (6,64%) — a energia cinética cresce com o quadrado da velocidade
- Rodovias residenciais aparecem como **protetoras** — limites de velocidade baixos reduzem a gravidade
- O LDA com sinal negativo no Stacking faz sentido: ele atua como "freio" para os modelos que tendem a superestimar a probabilidade de gravidade

**Overfitting foi observado?**
Não. Os resultados no teste são consistentes com o meta-holdout, sem quedas bruscas. As configurações conservadoras (XGBoost com 80 árvores, Random Forest com `max_samples=0.10`, MLP com `early_stopping=True`) colaboraram para isso.

---

## Como rodar esse notebook?

O notebook foi desenvolvido para rodar no **Google Colab** — sem necessidade de instalação local. Basta clicar no badge no topo desse README.

Se quiser rodar localmente, as principais dependências são:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib statsmodels shap
```

> TensorFlow é opcional — o notebook detecta automaticamente se está disponível e usa `MLPClassifier` do scikit-learn como fallback.

---

## Estrutura dos arquivos gerados

Ao executar o notebook, os seguintes arquivos são produzidos:

```
dataset/
├── dataset_amostra_limpa_binaria.parquet   # base limpa após outliers
├── features_selecionadas_binaria.json      # lista das 38 variáveis aprovadas pelo VIF
├── scaler_flag_pais.joblib                 # scaler treinado (StandardScaler)
├── limiares_decisao_flag_pais.json         # limiares ótimos por modelo
├── model_xgb_flag_pais.joblib              # modelo XGBoost treinado
├── model_stacking_flag_pais.joblib         # meta-aprendedor do Stacking
└── ...                                     # demais modelos e métricas

resultados_modelagem/
├── graficos_storytelling/                  # gráficos de EDA, ROC, PR, SHAP, t-SNE...
└── modelos_e_metricas/                     # matrizes de confusão, importância de features...
```

---

## Referências dos dados

- **Acidentes no Brasil (PRF):** https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf
- **Acidentes nos EUA (Kaggle):** https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
- **Infraestrutura viária (OpenStreetMap):** https://download.geofabrik.de/
- **Dados climáticos (ERA5/ECMWF):** https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
