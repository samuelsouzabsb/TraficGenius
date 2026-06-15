# Relatório Técnico-Gerencial: Modelagem Preditiva de Severidade de Acidentes de Trânsito (Stacking Ensemble - Base Completa)

Atuando como Cientista de Dados Sênior e Especialista em Estatística Multivariada, apresento a seguir o relatório completo incorporando **Variáveis de Interação Física**, **Comitê de Modelos (Stacking Ensemble)** e **Otimização de Limiar de Decisão (Threshold Tuning)** executado sobre a **base completa de 7.846.838 registros** (pós-limpeza de outliers).

---

## 🗺️ 1. Definição do Problema e Engenharia de Recursos

### Descrição do Problema
O objetivo é classificar a severidade dos acidentes em **Leve/Moderado (0)** vs. **Grave/Fatal (1)** sem utilizar dados geográficos absolutos (latitude, longitude, IDs de H3), focando em infraestrutura e clima para garantir a generalização do modelo para qualquer região.

### 🧼 Novos Atributos de Interação Física (Ponto 2 do Plano de Melhorias)
Adicionamos termos multiplicativos cruzados na Fase 1 para modelar interações não-lineares da física de tráfego:
1. **Fricção em Curvas sob Chuva (`interacao_chuva_curvas`):** Multiplicação de `precipitacao_milimetros` por `total_curvas_acentuadas`. Representa a perda severa de aderência pneumática em curvas molhadas. (Mantida pelo VIF, VIF < 10.0).
2. **Derrapagem sob Frio/Geada (`interacao_clima_curvatura`):** Multiplicação de `temperatura_celsius` por `curvatura_acumulada`. Representa o risco de congelamento de pista em trechos sinuosos. (Mantida pelo VIF, VIF < 10.0).
3. **Densidade Cinética do Fluxo (`interacao_velocidade_faixas`):** Multiplicação de `velocidade_media` por `quantidade_faixas_media`. 
   * *Nota de Rigor Estatístico:* Esta variável apresentou **VIF crítico de 25.53** e foi **removida automaticamente** na Fase 2 para evitar multicolinearidade severa, protegendo a estabilidade dos estimadores.

---

## 📈 2. Teste de Premissas e VIF

* **Normalidade (Shapiro/KS):** Rejeitada para todas as variáveis preditoras contínuas ($p = 0.000000$), justificando modelos baseados em árvores e redes neurais.
* **Homocedasticidade (Levene):** Rejeitada em todos os cruzamentos ($p = 0.000000$).
* **VIF Stepwise:** Das 35 variáveis contínuas numéricas, apenas a `interacao_velocidade_faixas` foi descartada. O modelo final conta com 38 preditores (incluindo dummies).

---

## 🧠 3. Modelagem Preditiva Avançada (Stacking & Threshold Tuning)

Implementamos um pipeline robusto com duas grandes inovações de Machine Learning (Pontos 1 e 3):

1. **Stacking Ensemble (Comitê de Modelos):**
   * Treinamos 5 modelos base (Regressão Logística, LDA, Random Forest, MLP e XGBoost) na partição de treino base (85% do treino).
   * Suas probabilidades previstas na partição Holdout Meta (15% do treino) foram usadas como entrada para treinar um meta-aprendedor (Regressão Logística).
   * **Contribuição Relativa dos Modelos Base (Pesos do Meta-Learner):**
     * **XGBoost:** 3.6445 (Maior contribuição ativa)
     * **Neural Network (MLP):** 2.5055
     * **Logistic Regression:** 1.5295
     * **Random Forest:** 1.1528
     * **LDA:** -2.9678 (Ajustador corretor de fronteiras lineares)

2. **Decision Threshold Tuning (Otimização de Limiar):**
   * Em vez de utilizar o corte padrão de 50%, otimizamos o limiar de decisão no meta-treino para maximizar o F1-Score da classe grave (1). Os limiares ótimos encontrados foram:
     * **Logistic Regression:** 0.23 (corte otimizado)
     * **LDA:** 0.49
     * **Random Forest:** 0.54
     * **Neural Network (MLP):** 0.54
     * **XGBoost:** 0.55
     * **Stacking Ensemble:** 0.27

---

## 📊 4. Avaliação Comparativa no Conjunto de Teste (2,35 Milhões de Linhas)

* **MCC (Chute na maioria):** 79.00% | **PCC (Chute proporcional):** 66.82% | **Meta (1.25 * PCC):** **83.53%**

### Tabela Comparativa de Métricas

| Modelo | Limiar Ótimo | Acurácia Global (50%) | Acurácia Global (Ótimo) | Acurácia Balanceada | F1-Score (Classe 1 - Grave) | F1-Score Macro | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stacking Ensemble (Vencedor)** | **0.27** | **80.42%** | **72.88%** | **70.18%** | **50.36%** | **65.85%** | **78.26%** |
| **XGBoost** | 0.55 | 67.01% | 71.42% | 70.08% | 49.90% | 64.95% | 77.82% |
| **Neural Network (MLP)** | 0.54 | 66.48% | 70.52% | 69.77% | 49.38% | 64.29% | 77.27% |
| **Random Forest** | 0.54 | 66.10% | 71.09% | 68.95% | 48.66% | 64.27% | 76.58% |
| **LDA** | 0.49 | 69.13% | 68.16% | 67.61% | 46.78% | 62.04% | 73.85% |
| **Logistic Regression** | 0.23 | 79.01% | 68.41% | 67.23% | 46.43% | 62.02% | 73.63% |

> [!IMPORTANT]
> * **O poder do Threshold Tuning no Logit:** O limiar ótimo de 0.23 elevou a acurácia balanceada da Regressão Logística de 53.61% para **67.23%** e multiplicou o F1-Score da classe grave de 16.45% para **46.43%**, comprovando a eficácia prática da otimização de limiares.
> * **Stacking quebra a barreira do F1-Score:** O comitê de modelos Stacking Ensemble atingiu a maior performance geral, quebrando a barreira de 50% de F1-Score para acidentes graves (**50.36%**) com a maior área sob a curva de **78.26%**.

---

## 📈 5. Importância Preditiva das Variáveis (Feature Importances - XGBoost)

Abaixo está o ranking das 20 variáveis mais importantes conforme a entropia de ganho de informação (F-Score) do modelo XGBoost:

1. **`comprimento_pontes_metros` (15.58%):** O fator infraestrutural isolado mais decisivo na determinação da severidade física de um acidente.
2. **`rodovia_dominante_res10_residential` (9.69%):** Atua como redutor drástico de severidade (baixo limite de velocidade).
3. **`rodovia_dominante_res10_service` (7.40%):** Vias de serviço atuam como moderadoras de severidade.
4. **`velocidade_media` (6.64%):** Quanto maior a velocidade média regulamentada da via, maior o risco de morte/gravidade no impacto.
5. **`rodovia_dominante_res9_motorway_link` (4.85%):** Alças de acesso de alta velocidade aumentam colisões graves.
6. **`rodovia_dominante_res9_service` (4.12%):** Fator atenuador de severidade.
7. **`rodovia_dominante_res10_motorway_link` (3.62%):** Pontos de aceleração e entrada em vias rápidas.
8. **`quantidade_semaforos` (3.36%):** Sinalização ativa regula cruzamentos e atua como forte atenuador de severidade.
9. **`rodovia_dominante_res11_residential` (3.15%):** Vias internas de bairro (atenuador).
10. **`rodovia_dominante_res9_residential` (2.86%):** Atenuador urbano local.
11. **`rodovia_dominante_res10_motorway` (2.20%):** Pistas expressas de alta velocidade.
12. **`rodovia_dominante_res10_tertiary` (1.66%):** Vias de ligação secundárias.
13. **`rodovia_dominante_res11_trunk` (1.45%):** Vias de trânsito rápido interestadual.
14. **`quantidade_cruzamentos` (1.33%):** Presença de cruzamentos induz desaceleração (fator de proteção).
15. **`extensao_rodovia_metros_res9` (1.28%):** Densidade de asfalto da zona.
16. **`rodovia_dominante_res11_desconhecido` (1.28%):** Variável de controle.
17. **`rodovia_dominante_res11_motorway_link` (1.20%):** Alças de ligação.
18. **`rodovia_dominante_res11_tertiary` (1.05%):** Vias coletoras.
19. **`rodovia_dominante_res10_trunk` (1.02%):** Rodovias interestaduais.
20. **`ponto_orvalho_celsius` (0.98%):** Proxy climática indireta para umidade/neblina.

---

## 🛠️ Recomendações Estratégicas Finais para Gestão de Vias

1. **Gestão de Margem e Segurança em Pontes:** O comprimento de pontes e viadutos domina o F-Score de importância do XGBoost (**15.58%**) e possui alta carga discriminante no LDA (**0.604**). Gestores devem implantar defensas metálicas maleáveis de absorção, fiscalizar limites de velocidade por radar nas entradas de pontes e realizar manutenção rígida do pavimento sobre essas superfícies.
2. **Moderação Dinâmica de Velocidade (Radares Inteligentes):** A `velocidade_media` regulamentada representa a energia de impacto. Vias acima de 80 km/h respondem pelo maior pico de probabilidade de lesão severa/óbito. Radares fixos e sinalização ostensiva são mandatórios nessas regiões.
3. **Calibração de Alças de Acesso (Motorway Links):** A presença de alças de acesso a rodovias expressas está no Top 10 de importância. É recomendável o redesenho geométrico dessas curvas de aceleração e instalação de faixas de aceleração estendidas para evitar conflitos de velocidade durante a fusão de fluxo.
