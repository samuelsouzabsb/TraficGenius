# Walkthrough: Avaliação dos Novos Treinamentos (Temporal, Países Separados e Flag de País)

Este walkthrough apresenta os resultados do treinamento e da avaliação de três novas configurações de modelos de classificação de severidade de acidentes de trânsito (Leve/Moderado vs. Grave/Fatal), comparando-as com o modelo baseline original.

---

## 🛠️ O que foi realizado

1. **Geração dos Novos Scripts de Treinamento**:
   - `[NEW]` [pipeline_fase3a5_modelagem_desde2020.py](file:///C:/Users/samue/Documents/trafic/pipeline_fase3a5_modelagem_desde2020.py): Treina os 5 modelos e Stacking utilizando apenas registros de 2020 em diante.
   - `[NEW]` [pipeline_fase3a5_modelagem_paises_separados.py](file:///C:/Users/samue/Documents/trafic/pipeline_fase3a5_modelagem_paises_separados.py): Separa os dados em US e BR e treina pipelines paralelos para cada país.
   - `[NEW]` [pipeline_fase3a5_modelagem_com_flag_pais.py](file:///C:/Users/samue/Documents/trafic/pipeline_fase3a5_modelagem_com_flag_pais.py): Treina um modelo unificado, utilizando o país (`pais_US` = 1 para EUA, 0 para Brasil) como uma variável preditora.
   - `[NEW]` [avaliar_novos_modelos.py](file:///C:/Users/samue/Documents/trafic/avaliar_novos_modelos.py): Consolida todas as métricas dos treinos, plota gráficos comparativos e grava relatórios consolidados.

2. **Treinamento e Validação**:
   - Todos os scripts foram executados com sucesso no conjunto de dados completo (7,8M de registros pós-limpeza).
   - Otimizamos o limiar de decisão para cada modelo para maximizar o F1-Score da classe grave (1).

---

## 📊 Comparativo Geral de Desempenho (Stacking Ensemble)

Abaixo estão os resultados do Stacking Ensemble (meta-aprendedor Logit) em cada uma das configurações avaliadas:

| Configuração | Limiar Ótimo | Acurácia Global (Ótimo) | Acurácia Balanceada | F1-Score Classe Grave (1) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** (Todo histórico, sem flag) | 0.27 | 72.88% | 70.18% | 50.36% | 78.26% |
| **Temporal (2020+)** | 0.20 | 76.50% | 68.32% | 39.40% | 76.71% |
| **EUA (Separado)** | 0.30 | 75.95% | 71.82% | **52.53%** | **80.19%** |
| **Brasil (Separado)** | 0.21 | 47.07% | 56.93% | 41.87% | 61.25% |
| **Flag de País** (Unificado com flag) | 0.27 | 72.87% | 70.61% | 50.80% | 78.66% |

---

## 📈 Gráfico Comparativo de Configurações

O gráfico abaixo compara o F1-Score da classe grave e o ROC-AUC de cada configuração de Stacking Ensemble:

![Comparativo Geral de Configurações](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/comparativo_geral_configuracoes.png)

---

## 🔍 Principais Insights Estatísticos e Conclusões

### 1. Desempenho EUA vs. Brasil (Países Separados)
* **EUA (Separado)** obteve o **melhor F1-Score (52.53%) e ROC-AUC (80.19%)** de todas as configurações. Isso indica que a dinâmica de acidentes nos EUA possui alta consistência estatística com as features físicas coletadas (como pontes, velocidade média urbana e clima).
* **Brasil (Separado)** teve o pior desempenho preditivo (ROC-AUC de **61.25%** e F1-Score de **41.87%**). Isso ocorre devido ao fato de os dados de acidentes brasileiros possuírem menor calibração em relação aos atributos de infraestrutura coletados das rodovias dominantes ou devido ao alto desbalanceamento (75% / 25%) sob menor amostragem absoluta.
* *Implicação*: Treinar um modelo específico para o Brasil requer features adicionais ou amostragem diferenciada.

### 2. Inclusão da Flag de País no Modelo Unificado
* O modelo **Flag de País** alcançou um ganho marginal em relação à Baseline: F1-Score subiu de **50.36% para 50.80%** e ROC-AUC de **78.26% para 78.66%**.
* No ranking de importância do XGBoost (Top 15), a feature `pais_US` obteve **3.83% de importância relativa**, superando variáveis tradicionais como o `Mes` de ocorrência e cruzamentos. Isso confirma que a localização nacional tem um peso de calibragem estrutural relevante para alinhar as duas bases.

### 3. Corte Temporal pós-2020
* Restringir o dataset aos anos 2020-2026 resultou em uma queda significativa de desempenho (F1-Score caiu de **50.36% para 39.40%**).
* Isso sugere que o volume histórico pré-2020 é essencial para o aprendizado das dinâmicas raras de acidentes graves, e que o padrão físico das vias e clima não se alterou o suficiente a ponto de justificar o descarte de histórico mais antigo em favor da temporalidade pura.

---

## 🗂️ Localização dos Artefatos Gerados

- **Relatório Completo Detalhado**: [relatorio_comparativo_final.md](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/relatorio_comparativo_final.md)
- **Gráfico Comparativo Geral**: [comparativo_geral_configuracoes.png](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/comparativo_geral_configuracoes.png)
- **Tabelas de Métricas em CSV**:
  - `dataset/comparativo_modelos_desde2020.csv`
  - `dataset/comparativo_modelos_us.csv`
  - `dataset/comparativo_modelos_br.csv`
  - `dataset/comparativo_modelos_flag_pais.csv`
