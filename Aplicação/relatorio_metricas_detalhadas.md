# Relatório Estatístico Detalhado das Métricas dos Modelos

Este relatório descreve detalhadamente o desempenho de cada um dos quatro modelos treinados (XGBoost, Random Forest, Regressão Logística e CNN 1D).

## Modelo: XGBoost

### 1. Matriz de Confusão
A matriz mostra as predições nas colunas e os valores reais nas linhas (G1 a G4):

| Real \ Predito | G1 (Leve) | G2 (Moderado) | G3 (Grave) | G4 (Fatal) |
| :--- | :---: | :---: | :---: | :---: |
| **G1** | 461 | 13 | 41 | 15 |
| **G2** | 4,758 | 22,334 | 8,874 | 11,795 |
| **G3** | 1,139 | 847 | 6,743 | 1,394 |
| **G4** | 35 | 234 | 57 | 1,260 |


### 2 e 3. Métricas por Classe (F1-Score e Recall)
| Classe de Severidade | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: |
| **G1** | 86.98% | 13.32% |
| **G2** | 46.76% | 62.75% |
| **G3** | 66.61% | 52.19% |
| **G4** | 79.45% | 15.70% |


### 4. Valores de ROC-AUC (One-vs-Rest)
A abordagem utilizada é a **One-vs-Rest (OVR)**, que avalia cada classe em relação a todas as outras reunidas. Apresentamos abaixo os valores por classe, a média macro e a média ponderada (weighted):

| Tipo de ROC-AUC (OVR) | Valor (%) |
| :--- | :---: |
| **G1 vs Rest** | 94.90% |
| **G2 vs Rest** | 78.41% |
| **G3 vs Rest** | 84.08% |
| **G4 vs Rest** | 85.78% |
| **Macro ROC-AUC (Média Simples)** | **85.79%** |
| **Weighted ROC-AUC (Média Ponderada por Suporte)** | **79.71%** |

---

## Modelo: Random Forest

### 1. Matriz de Confusão
A matriz mostra as predições nas colunas e os valores reais nas linhas (G1 a G4):

| Real \ Predito | G1 (Leve) | G2 (Moderado) | G3 (Grave) | G4 (Fatal) |
| :--- | :---: | :---: | :---: | :---: |
| **G1** | 431 | 19 | 65 | 15 |
| **G2** | 5,011 | 18,231 | 10,753 | 13,766 |
| **G3** | 1,407 | 820 | 6,444 | 1,452 |
| **G4** | 30 | 285 | 56 | 1,215 |


### 2 e 3. Métricas por Classe (F1-Score e Recall)
| Classe de Severidade | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: |
| **G1** | 81.32% | 11.63% |
| **G2** | 38.17% | 54.33% |
| **G3** | 63.66% | 46.97% |
| **G4** | 76.61% | 13.47% |


### 4. Valores de ROC-AUC (One-vs-Rest)
A abordagem utilizada é a **One-vs-Rest (OVR)**, que avalia cada classe em relação a todas as outras reunidas. Apresentamos abaixo os valores por classe, a média macro e a média ponderada (weighted):

| Tipo de ROC-AUC (OVR) | Valor (%) |
| :--- | :---: |
| **G1 vs Rest** | 92.63% |
| **G2 vs Rest** | 73.78% |
| **G3 vs Rest** | 78.63% |
| **G4 vs Rest** | 81.67% |
| **Macro ROC-AUC (Média Simples)** | **81.68%** |
| **Weighted ROC-AUC (Média Ponderada por Suporte)** | **74.97%** |

---

## Modelo: Regressão Logística

### 1. Matriz de Confusão
A matriz mostra as predições nas colunas e os valores reais nas linhas (G1 a G4):

| Real \ Predito | G1 (Leve) | G2 (Moderado) | G3 (Grave) | G4 (Fatal) |
| :--- | :---: | :---: | :---: | :---: |
| **G1** | 393 | 27 | 73 | 37 |
| **G2** | 10,961 | 12,110 | 14,015 | 10,675 |
| **G3** | 1,901 | 1,143 | 5,147 | 1,932 |
| **G4** | 223 | 208 | 363 | 792 |


### 2 e 3. Métricas por Classe (F1-Score e Recall)
| Classe de Severidade | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: |
| **G1** | 74.15% | 5.61% |
| **G2** | 25.36% | 39.54% |
| **G3** | 50.84% | 34.64% |
| **G4** | 49.94% | 10.54% |


### 4. Valores de ROC-AUC (One-vs-Rest)
A abordagem utilizada é a **One-vs-Rest (OVR)**, que avalia cada classe em relação a todas as outras reunidas. Apresentamos abaixo os valores por classe, a média macro e a média ponderada (weighted):

| Tipo de ROC-AUC (OVR) | Valor (%) |
| :--- | :---: |
| **G1 vs Rest** | 82.58% |
| **G2 vs Rest** | 60.57% |
| **G3 vs Rest** | 65.09% |
| **G4 vs Rest** | 70.76% |
| **Macro ROC-AUC (Média Simples)** | **69.75%** |
| **Weighted ROC-AUC (Média Ponderada por Suporte)** | **61.80%** |

---
