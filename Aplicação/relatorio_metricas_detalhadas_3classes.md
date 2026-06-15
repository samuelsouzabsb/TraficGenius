# Relatório Estatístico Detalhado das Métricas dos Modelos (3 Classes)

Este relatório descreve detalhadamente o desempenho de cada um dos modelos treinados com 3 classes de severidade (Leve/Médio, Grave e Fatal).

## Modelo: XGBoost

### 1. Matriz de Confusão
A matriz mostra as predições nas colunas e os valores reais nas linhas:

| Real \ Predito | Leve/Médio | Grave | Fatal |
| :--- | :---: | :---: | :---: |
| **Leve/Médio** | 25,623 | 11,274 | 11,394 |
| **Grave** | 953 | 7,853 | 1,317 |
| **Fatal** | 242 | 75 | 1,269 |


### 2 e 3. Métricas por Classe (F1-Score e Recall)
| Classe de Severidade | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: |
| **Leve/Médio** | 53.06% | 68.23% |
| **Grave** | 77.58% | 53.56% |
| **Fatal** | 80.01% | 16.30% |


### 4. Valores de ROC-AUC (One-vs-Rest)
| Tipo de ROC-AUC (OVR) | Valor (%) |
| :--- | :---: |
| **Leve/Médio vs Rest** | 81.61% |
| **Grave vs Rest** | 85.64% |
| **Fatal vs Rest** | 86.27% |
| **Macro ROC-AUC (Média Simples)** | **84.51%** |
| **Weighted ROC-AUC (Média Ponderada por Suporte)** | **82.42%** |

---

## Modelo: Random Forest

### 1. Matriz de Confusão
A matriz mostra as predições nas colunas e os valores reais nas linhas:

| Real \ Predito | Leve/Médio | Grave | Fatal |
| :--- | :---: | :---: | :---: |
| **Leve/Médio** | 21,016 | 14,044 | 13,231 |
| **Grave** | 897 | 7,875 | 1,351 |
| **Fatal** | 292 | 71 | 1,223 |


### 2 e 3. Métricas por Classe (F1-Score e Recall)
| Classe de Severidade | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: |
| **Leve/Médio** | 43.52% | 59.62% |
| **Grave** | 77.79% | 49.05% |
| **Fatal** | 77.11% | 14.06% |


### 4. Valores de ROC-AUC (One-vs-Rest)
| Tipo de ROC-AUC (OVR) | Valor (%) |
| :--- | :---: |
| **Leve/Médio vs Rest** | 75.92% |
| **Grave vs Rest** | 80.92% |
| **Fatal vs Rest** | 82.11% |
| **Macro ROC-AUC (Média Simples)** | **79.65%** |
| **Weighted ROC-AUC (Média Ponderada por Suporte)** | **76.93%** |

---

## Modelo: Regressão Logística

### 1. Matriz de Confusão
A matriz mostra as predições nas colunas e os valores reais nas linhas:

| Real \ Predito | Leve/Médio | Grave | Fatal |
| :--- | :---: | :---: | :---: |
| **Leve/Médio** | 18,772 | 16,916 | 12,603 |
| **Grave** | 1,759 | 6,156 | 2,208 |
| **Fatal** | 285 | 409 | 892 |


### 2 e 3. Métricas por Classe (F1-Score e Recall)
| Classe de Severidade | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: |
| **Leve/Médio** | 38.87% | 54.33% |
| **Grave** | 60.81% | 36.64% |
| **Fatal** | 56.24% | 10.32% |


### 4. Valores de ROC-AUC (One-vs-Rest)
| Tipo de ROC-AUC (OVR) | Valor (%) |
| :--- | :---: |
| **Leve/Médio vs Rest** | 63.88% |
| **Grave vs Rest** | 67.46% |
| **Fatal vs Rest** | 70.83% |
| **Macro ROC-AUC (Média Simples)** | **67.39%** |
| **Weighted ROC-AUC (Média Ponderada por Suporte)** | **64.66%** |

---
