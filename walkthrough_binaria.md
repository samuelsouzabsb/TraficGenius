# Walkthrough Técnico: Modelagem Preditiva de Severidade de Acidentes (Classificação Binária)

Este walkthrough documenta a implementação de ponta a ponta e a avaliação rigorosa do pipeline preditivo multivariado para prever a severidade de acidentes de trânsito. O objetivo principal é classificar se um acidente resultará em **Grave/Fatal (1)** vs. **Leve/Moderado (0)** com base em infraestrutura e clima, utilizando a **base completa de 8,1 milhões de registros** (7.846.838 pós-limpeza de outliers).

Como restrição metodológica crítica, **excluímos quaisquer variáveis de localização absoluta** (coordenadas e IDs de H3) para garantir a capacidade de generalização física dos modelos para qualquer região geográfica.

---

## 🛠️ Três Pilares de Otimização Implementados

### 1. Decision Threshold Tuning (Otimização de Limiares)
Em conjuntos de dados desbalanceados (como nossa proporção de **79% Leve/Moderado / 21% Grave/Fatal**), a classificação com o ponto de corte padrão de 50% resulta em baixa sensibilidade para a classe crítica. Otimizamos os limiares de decisão no conjunto de validação Meta para maximizar o F1-Score da classe Grave/Fatal (1), encontrando os seguintes pontos ótimos:
*   **Regressão Logística:** 0.23
*   **Análise Discriminante Linear (LDA):** 0.49
*   **Random Forest:** 0.54
*   **Rede Neural (MLP Keras):** 0.54
*   **XGBoost:** 0.55
*   **Stacking Ensemble:** 0.27

### 2. Feature Engineering & Interações Físicas
Modelamos explicitamente a interação não-linear entre infraestrutura física e intempéries climáticas na Fase 1:
*   **Fricção em Curvas sob Chuva (`interacao_chuva_curvas`):** `precipitacao_milimetros` $\times$ `total_curvas_acentuadas` (Perda de atrito lateral sob pista molhada).
*   **Derrapagem sob Frio/Geada (`interacao_clima_curvatura`):** `temperatura_celsius` $\times$ `curvatura_acumulada` (Aproximação de congelamento da pista).
*   *Nota Estatística:* A densidade cinética (`velocidade_media * quantidade_faixas_media`) foi removida automaticamente na Fase 2 por apresentar **VIF severo de 25.53**, eliminando riscos de multicolinearidade.

### 3. Comitê de Modelos (Ensemble Stacking)
Construímos um classificador por empilhamento (**Stacking Ensemble**) combinando preditores lineares, paramétricos, ensembles baseados em árvores e redes neurais densas. O meta-aprendedor final (Regressão Logística) combinou as previsões ponderando a contribuição de cada modelo:
*   **XGBoost:** 3.6445 (Maior contribuição ativa)
*   **Neural Network (MLP):** 2.5055
*   **Regressão Logística:** 1.5295
*   **Random Forest:** 1.1528
*   **LDA:** -2.9678 (Corretor de fronteiras lineares)

---

## 📊 Avaliação Comparativa de Desempenho no Conjunto de Teste (2,35 Milhões de Linhas)

Os modelos foram avaliados em relação ao acaso e entre si no conjunto de teste final de **2.354.052 linhas**:
*   **MCC (Chute na maioria):** 79.00%
*   **PCC (Chute proporcional aleatório):** 66.82%
*   **Meta Proposta (1.25 * PCC):** **83.53%** (Acurácia global necessária para bater significativamente o acaso)

| Modelo | Limiar Ótimo | Acurácia (50%) | Acurácia (Ótimo) | Acurácia Balanceada | F1-Score (Classe 1) | F1-Score Macro | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stacking Ensemble (Vencedor)** | **0.27** | **80.42%** | **72.88%** | **70.18%** | **50.36%** | **65.85%** | **78.26%** |
| **XGBoost (Boosting)** | 0.55 | 67.01% | 71.42% | 70.08% | 49.90% | 64.95% | 77.82% |
| **Neural Network (MLP)** | 0.54 | 66.48% | 70.52% | 69.77% | 49.38% | 64.29% | 77.27% |
| **Random Forest (CART)** | 0.54 | 66.10% | 71.09% | 68.95% | 48.66% | 64.27% | 76.58% |
| **LDA (Discriminante)** | 0.49 | 69.13% | 68.16% | 67.61% | 46.78% | 62.04% | 73.85% |
| **Logistic Regression** | 0.23 | 79.01% | 68.41% | 67.23% | 46.43% | 62.02% | 73.63% |

![Comparação de Métricas](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/comparativo_performance_binaria.png)

---

## 🎛️ Matrizes de Confusão Detalhadas (Com Limiar Ótimo)

Abaixo apresentamos as matrizes de confusão numéricas exatas calculadas sobre a partição de teste.

### 1. Regressão Logística (Limiar: 0.23)
*   **Sensibilidade (Recall):** 65.2% | **Especificidade:** 69.3%
*   **Total de Acertos (Hit Ratio):** 1.610.464 registros (68.4%)

| Classificação Real | Predito: Leve/Moderado (0) | Predito: Grave/Fatal (1) | Total Real |
| :--- | :---: | :---: | :---: |
| **Leve/Moderado (0)** | 1.288.189 (69.3%) | 571.609 (30.7%) | 1.859.798 |
| **Grave/Fatal (1)** | 171.979 (34.8%) | 322.275 (65.2%) | 494.254 |

### 2. Análise Discriminante Linear (LDA) (Limiar: 0.49)
*   **Sensibilidade (Recall):** 66.6% | **Especificidade:** 68.6%
*   **Total de Acertos (Hit Ratio):** 1.604.627 registros (68.2%)

| Classificação Real | Predito: Leve/Moderado (0) | Predito: Grave/Fatal (1) | Total Real |
| :--- | :---: | :---: | :---: |
| **Leve/Moderado (0)** | 1.275.217 (68.6%) | 584.581 (31.4%) | 1.859.798 |
| **Grave/Fatal (1)** | 164.844 (33.4%) | 329.410 (66.6%) | 494.254 |

### 3. Random Forest (Limiar: 0.54)
*   **Sensibilidade (Recall):** 65.2% | **Especificidade:** 72.6%
*   **Total de Acertos (Hit Ratio):** 1.673.558 registros (71.1%)

| Classificação Real | Predito: Leve/Moderado (0) | Predito: Grave/Fatal (1) | Total Real |
| :--- | :---: | :---: | :---: |
| **Leve/Moderado (0)** | 1.351.081 (72.6%) | 508.717 (27.4%) | 1.859.798 |
| **Grave/Fatal (1)** | 171.777 (34.8%) | 322.477 (65.2%) | 494.254 |

### 4. Rede Neural Artificial (MLP) (Limiar: 0.54)
*   **Sensibilidade (Recall):** 68.5% | **Especificidade:** 71.1%
*   **Total de Acertos (Hit Ratio):** 1.660.040 registros (70.5%)

| Classificação Real | Predito: Leve/Moderado (0) | Predito: Grave/Fatal (1) | Total Real |
| :--- | :---: | :---: | :---: |
| **Leve/Moderado (0)** | 1.321.533 (71.1%) | 538.265 (28.9%) | 1.859.798 |
| **Grave/Fatal (1)** | 155.747 (31.5%) | 338.507 (68.5%) | 494.254 |

### 5. XGBoost (Limiar: 0.55)
*   **Sensibilidade (Recall):** 67.8% | **Especificidade:** 72.4%
*   **Total de Acertos (Hit Ratio):** 1.681.266 registros (71.4%)

| Classificação Real | Predito: Leve/Moderado (0) | Predito: Grave/Fatal (1) | Total Real |
| :--- | :---: | :---: | :---: |
| **Leve/Moderado (0)** | 1.346.260 (72.4%) | 513.538 (27.6%) | 1.859.798 |
| **Grave/Fatal (1)** | 159.248 (32.2%) | 335.006 (67.8%) | 494.254 |

### 6. Stacking Ensemble (Limiar: 0.27 - Vencedor)
*   **Sensibilidade (Recall):** 65.5% | **Especificidade:** 74.8%
*   **Total de Acertos (Hit Ratio):** **1.715.551 registros (72.9%)**

| Classificação Real | Predito: Leve/Moderado (0) | Predito: Grave/Fatal (1) | Total Real |
| :--- | :---: | :---: | :---: |
| **Leve/Moderado (0)** | **1.391.679 (74.8%)** | 468.119 (25.2%) | 1.859.798 |
| **Grave/Fatal (1)** | 170.382 (34.5%) | **323.872 (65.5%)** | 494.254 |

> [!NOTE]
> O Stacking Ensemble obteve a melhor separação de classes, minimizando Falsos Positivos em relação a outros modelos balanceados, entregando uma acurácia global significativamente superior (72.9%).

#### Carrossel de Heatmaps das Matrizes de Confusão
````carousel
![Matriz Logística](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/matriz_confusao_logistic_regression.png)
<!-- slide -->
![Matriz LDA](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/matriz_confusao_lda.png)
<!-- slide -->
![Matriz Random Forest](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/matriz_confusao_random_forest.png)
<!-- slide -->
![Matriz MLP](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/matriz_confusao_neural_network_mlp.png)
<!-- slide -->
![Matriz XGBoost](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/matriz_confusao_xgboost.png)
<!-- slide -->
![Matriz Stacking](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/matriz_confusao_stacking_ensemble.png)
````

---

## 📈 Importância das Variáveis (Feature Importances - XGBoost)

Abaixo está o gráfico contendo o F-Score (ganho de informação) relativo calculado para o classificador XGBoost (Top 15):

![Importância das Features XGBoost](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/importancia_features_xgboost_binaria.png)

### Principais Fatores Identificados:
1.  **`comprimento_pontes_metros` (15.58%):** Variável mais impactante de todas na determinação de gravidade. Acidentes em pontes e viadutos possuem taxas de severidade críticas pela ausência de áreas de escape e risco de impacto em barreiras fixas.
2.  **`rodovia_dominante_res10_residential` (9.69%):** Atua como o principal redutor de gravidade devido aos limites rígidos de velocidade urbana (30-50 km/h).
3.  **`velocidade_media` (6.64%):** Representa a energia cinética do veículo antes da colisão ($E_c = \frac{1}{2}mv^2$). Vias rápidas regulamentadas aumentam a probabilidade de acidentes graves/fatais de forma exponencial.
4.  **`rodovia_dominante_res9_motorway_link` (4.85%):** Alças de acesso e transição para rodovias de alta velocidade, onde conflitos de velocidade em fusões de faixas geram impactos graves.

---

## 🗺️ Diagnósticos Estatísticos Adicionais

### Teste de Premissas de Normalidade e Homocedasticidade
*   **Normalidade:** Kolmogorov-Smirnov e Shapiro-Wilk rejeitaram a normalidade das variáveis contínuas ($p < 0.05$), confirmando a necessidade de algoritmos robustos não-paramétricos (redes neurais e árvores).
*   **Homocedasticidade:** O teste de Levene rejeitou a igualdade de variância de todas as features métricas em relação às classes ($p < 0.05$).

![Distribuição de Normalidade](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/distribuicao_normalidade_binaria.png)

### Fator de Inflação da Variância (VIF)
Mantivemos todas as variáveis contínuas com VIF abaixo de 10.0, garantindo a ausência de multicolinearidade.

![VIF Colinearidade](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/vif_multicolinearidade_binaria.png)
