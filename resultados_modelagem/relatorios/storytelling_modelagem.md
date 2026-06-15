# Relatório Técnico: Storytelling Estatístico e Diagnósticos da Modelagem

Este relatório apresenta a análise científica e o storytelling estatístico completo para a modelagem preditiva de severidade de acidentes rodoviários (Leve/Moderado vs. Grave/Fatal), utilizando o modelo final unificado com **Flag de País** (contendo a feature `pais_US` para calibração conjunta da base americana e brasileira).

O objetivo é expor com rigor acadêmico a aderência dos dados às premissas clássicas, o comportamento exploratório multivariado e a capacidade discriminante e interpretativa da arquitetura preditiva desenvolvida.

---

## 📊 Visualizações e Resultados de Storytelling

Abaixo estão agrupados os gráficos premium gerados no diretório de artefatos da conversa, divididos nas etapas lógicas de modelagem:

````carousel
### Análise Exploratória (EDA)
Visualizações que retratam a dispersão e perfis das features físicas:

- **Densidades Kernel (KDE)**: [KDE Distribuição por Severidade](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/story_eda_kde.png)
- **Boxplots Comparativos**: [Dispersão de Variáveis por Severidade](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/story_eda_boxplots.png)
- **Matriz de Correlação**: [Correlação Linear do Top 10 Preditores](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/story_eda_correlation.png)
- **Coordenadas Paralelas**: [Perfis Multivariados dos Acidentes](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/story_eda_parallel.png)

![KDE Distribuição](C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/story_eda_kde.png)
<!-- slide -->
### Validação de Premissas Clássicas
Gráficos de comportamento de resíduos e multicolinearidade para modelos clássicos:

- **Normal Q-Q Plot (NPP)**: [Gráfico de Probabilidade Normal dos Resíduos](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/story_premissas_qqplot.png)
- **Resíduos vs. Valores Ajustados**: [Verificação de Homocedasticidade](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/story_premissas_residuals_fitted.png)
- **Correlograma ACF**: [Autocorrelação Sequencial dos Erros](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/story_premissas_acf.png)
- **Bar Chart de VIF**: [Fatores de Inflação da Variância por Feature](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/story_premissas_vif.png)

![QQ-Plot](C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/story_premissas_qqplot.png)
<!-- slide -->
### Desempenho Preditivo e Explicações SHAP
Curvas avançadas de ajuste, discriminação, e explicabilidade do Stacking Ensemble e XGBoost:

- **Curvas ROC**: [Taxa de Falsos vs. Verdadeiros Positivos](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/story_desempenho_roc.png)
- **Precision-Recall (PR)**: [Acurácia de Classificação de Acidentes Graves](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/story_desempenho_pr.png)
- **Curva de Calibração**: [Reliability Diagram de Probabilidades](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/story_desempenho_calibration.png)
- **Lift & Ganho Acumulado**: [Utilidade Prática da Priorização](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/story_desempenho_lift_gain.png)
- **Projeção t-SNE 2D**: [Fronteira de Classificação Espacial](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/story_desempenho_tsne.png)
- **SHAP Summary Plot**: [Tree SHAP Feature Importance no XGBoost](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/story_desempenho_shap.png)

![Curva ROC](C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/story_desempenho_roc.png)
````

---

## 1. Análise de Outliers (Univariados e Multivariados)

A identificação de anomalias nos dados foi realizada por meio de Z-Scores (tradicional e modificado pelo desvio absoluto da mediana - MAD) para as principais features contínuas físicas e ambientais, além da **Distância de Mahalanobis ($D^2$)** para avaliar outliers multivariados estruturais.

### Resultados de Outliers Univariados
*   **Velocidade Média**:
    *   Média: $78.88 \text{ km/h}$ | Mediana: $77.76 \text{ km/h}$ | MAD: $13.39 \text{ km/h}$
    *   Outliers Z-Score: **0.01%** | Outliers Mod. Z-Score (MAD): **0.01%**
*   **Temperatura (Celsius)**:
    *   Média: $16.21 ^\circ\text{C}$ | Mediana: $17.52 ^\circ\text{C}$ | MAD: $6.90 ^\circ\text{C}$
    *   Outliers Z-Score: **0.21%** | Outliers Mod. Z-Score (MAD): **0.22%**
*   **Precipitação (Milímetros)**:
    *   Média: $0.11 \text{ mm}$ | Mediana: $0.0 \text{ mm}$ | MAD: $1\times 10^{-5} \text{ mm}$
    *   Outliers Z-Score: **2.31%** | Outliers Mod. Z-Score (MAD): **39.03%**
    *   *Nota*: A alta percentagem no Z-Score Modificado se deve ao fato de que a precipitação tem mediana $0.0$, o que achata o MAD e artificialmente infla o corte modificado em variáveis com excesso de zeros (zero-inflated).

### Diagnóstico de Outliers Multivariados
Calculando a **Distância de Mahalanobis ($D^2$)** no espaço multidimensional das 28 features contínuas:
*   Graus de Liberdade (DF): $28$
*   Valor Crítico ($\chi^2$ a $99.9\%$ de confiança): **$56.89$**
*   Distância Máxima de Mahalanobis encontrada: $20,406.10$
*   Proporção de Outliers Multivariados ($D^2 > 56.89$): **$6.63\%$**

> [!NOTE]
> Enquanto a análise univariada sugere que menos de 1% dos dados são anômalos isoladamente, a análise multivariada detecta que $6.63\%$ dos registros combinam atributos de forma atípica (ex: alta velocidade em trechos de altíssima curvatura sob congelamento), o que justifica o emprego de modelos de aprendizado de máquina robustos que toleram interações não-lineares complexas.

---

## 2. Testes de Premissas dos Modelos Clássicos

Para validar as suposições estatísticas clássicas de modelos de regressão linear e logística ordinária (OLS e GLM), extraímos os resíduos de uma regressão auxiliar ajustada em amostras representativas dos dados de treino:

### 2.1 Aderência à Normalidade dos Resíduos
*   **Assimetria (Skewness)**: $1.17$ (Indica assimetria positiva moderada a alta).
*   **Curtose (Kurtosis)**: $0.07$ (Distribuição ligeiramente mais achatada que a normal teórica).
*   **Teste de Jarque-Bera**: Estatística $11,464.93$ | **p-valor = 0.00**
*   **Teste de Shapiro-Wilk (N=5000)**: Estatística $0.820$ | **p-valor = $1.32\times 10^{-59}$**
*   **Teste de Kolmogorov-Smirnov**: Estatística $0.258$ | **p-valor = 0.00**

> [!WARNING]
> **Rejeição Crítica da Normalidade**: Todos os testes estatísticos rejeitam formalmente a hipótese de normalidade dos resíduos ($p < 0.01$). Isso se deve ao fato de o fenômeno estudado ser binário (classificação de severidade de acidentes), o que viola intrinsicamente a normalidade residual linear e demonstra a inadequação de modelos puramente lineares que assumem normalidade.

### 2.2 Homocedasticidade (Igualdade de Variâncias)
*   **Teste de Breusch-Pagan-Godfrey (BPG)**:
    *   Estatística LM: $6,015.11$ | **p-valor = 0.00**
    *   Estatística F: $71.84$ | **p-valor = 0.00**
*   **Teste de Levene (velocidade_media entre severidades 0 e 1)**:
    *   Estatística W: $8.86$ | **p-valor = 0.0029**

> [!IMPORTANT]
> **Presença de Heterocedasticidade**: A variação dos resíduos não é constante ao longo do espaço amostral. O p-valor de Breusch-Pagan é zero absoluto. O Teste de Levene para `velocidade_media` confirma que a variância dessa feature se altera de forma estatisticamente significante entre acidentes leves/moderados e graves/fatais ($p < 0.01$). Isso exige o uso de algoritmos de árvore ou modelos baseados em gradiente que não sofrem perda de eficiência com variância não-constante.

### 2.3 Independência dos Erros (Ausência de Autocorrelação)
*   **Estatística de Durbin-Watson**: **$1.997$** (Extremamente próxima de $2.00$, indicando independência perfeita).
*   **Runs Test (Teste das Carreiras para Independência)**:
    *   Estatística Z: $-0.027$ | **p-valor = 0.979**

> [!TIP]
> **Independência Confirmada**: O p-valor do Runs Test é $0.979$ (não significativo), o que confirma que não há autocorrelação serial ou sequencial sistemática nos resíduos. A ordem dos acidentes inserida no dataset é independente do erro residual, satisfazendo a suposição de observações identicamente distribuídas e independentes (i.i.d.).

### 2.4 Ausência de Multicolinearidade
*   **Fatores de Inflação da Variância (VIF)**:
    *   Todas as features contínuas apresentaram VIF inferior a **$4.25$** (O VIF máximo foi obtido em `ponto_orvalho_celsius` com $4.25$, seguido por `temperatura_celsius` com $3.96$).
*   **Número de Condição (Condition Index)**:
    *   Índice de Condição Máximo: **$4.32$** (Valores abaixo de $10.0$ mostram colinearidade conjunta fraca e insignificante).

> [!TIP]
> **Multicolinearidade Inexistente**: Como todos os VIFs estão confortavelmente abaixo de $10.0$ (e mesmo de $5.0$) e o Condition Index está bem abaixo de $15.0$, a estabilidade matemática do modelo está garantida. Não há redundância prejudicial entre os preditores ambientais e físicos.

### 2.5 Linearidade e Especificação
*   **Ramsey RESET Test**:
    *   Estatística F: $36.43$ | **p-valor = $1.23\times 10^{-8}$**

> [!CAUTION]
> **Erro de Especificação Linear**: O RESET de Ramsey rejeita fortemente a especificação linear ($p < 0.01$). Isso demonstra de forma incontestável que existem termos não-lineares, interações quadráticas e de maior ordem cruciais omitidos em uma especificação puramente aditiva linear. Isso consolida e justifica cientificamente a escolha de **XGBoost, Random Forest e Redes Neurais (MLP)** como a abordagem ideal em detrimento da Regressão Logística tradicional.

---

## 3. Avaliação de Desempenho Geral e Storytelling das Curvas

Com base no modelo de **Flag de País** avaliado no conjunto de teste:

*   **Curvas ROC**: O Stacking Ensemble e o XGBoost dominam o gráfico com ROC-AUC de **$78.66\%$** e **$78.34\%$**, respectivamente. A Regressão Logística e a LDA servem de baselines de calibração lineares.
*   **Curvas Precision-Recall (PR)**: O baseline proporcional de acidentes graves na amostra é de $24.7\%$. A curva de PR demonstra que ambos, XGBoost e Stacking, conseguem manter precisões acima de $55\%$ mesmo em taxas de cobertura (Recall) superiores a $60\%$.
*   **Calibração (Reliability Diagram)**: A curva de calibração mostra que o Stacking Ensemble (meta-aprendedor Logit) e a Regressão Logística estão perfeitamente alinhados à diagonal de calibração ideal. O XGBoost, embora tenha ótima discriminação, apresenta uma leve subestimação de probabilidades em faixas intermediárias que foi corrigida pelo meta-aprendedor do Stacking.
*   **Lift & Ganho Acumulado**:
    *   Ao ordenar os dados pelas maiores probabilidades do Stacking Ensemble, os **Top 20%** de acidentes priorizados pelo modelo contêm **58%** de todos os acidentes graves reais do dataset (um Lift de **2.9x** sobre um modelo aleatório).
    *   Atingir 80% de detecção de todos os acidentes graves exige inspecionar apenas os **Top 42%** das maiores previsões de probabilidade do modelo.
*   **Projeção t-SNE 2D**: O plot em duas dimensões revela agrupamentos bem delineados no espaço reduzido de atributos físicos-ambientais. A fronteira de decisão separa as nuvens de pontos de severidade de forma clara, embora com sobreposições marginais que representam a aleatoriedade inerente ao tráfego rodoviário.

---

## 4. Explicações Globais com SHAP (XGBoost)

O gráfico SHAP Summary Plot revela a importância de contribuição marginal exata de cada preditor na tomada de decisão de severidade do XGBoost:

1.  **`velocidade_media`**: O preditor com maior força de decisão. Altas velocidades (valores vermelhos) empurram as previsões fortemente para a classe grave (SHAP value altamente positivo).
2.  **`rodovia_dominante_res11` / `res10`**: A infraestrutura espacial física da rodovia dominante desempenha o segundo maior peso. Segmentos com alta densidade de rodovias conectadas atuam como moderadores ou amplificadores dependendo de suas taxas de conservação.
3.  **`pais_US` (Flag de País)**: A feature de controle nacional está no Top 10 de importância. A localidade nacional americana atua recalibrando o intercepto base, capturando a diferença estrutural nas estatísticas nacionais e taxas de gravidade entre os dois sistemas de tráfego.
4.  **`temperatura_celsius`**: Temperaturas baixas (valores azuis) aumentam a probabilidade de acidentes graves devido à possibilidade de congelamento da pista e redução de aderência.
5.  **`quantidade_semaforos` e `quantidade_cruzamentos`**: Menor densidade desses dispositivos em rodovias rápidas está associada a acidentes mais graves (devido à velocidade operacional superior permitida em trechos livres).

---

## 💾 Localização e Consistência dos Arquivos

Os arquivos gerados estão armazenados nos caminhos corretos e prontos para uso:
*   Métricas e Diagnósticos (JSON): [diagnosticos_estatisticos.json](file:///C:/Users/samue/Documents/trafic/dataset/diagnosticos_estatisticos.json)
*   Imagens de Storytelling (Pasta de Artefatos):
    *   EDA KDE: [story_eda_kde.png](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/story_eda_kde.png)
    *   EDA Boxplots: [story_eda_boxplots.png](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/story_eda_boxplots.png)
    *   EDA Parallel: [story_eda_parallel.png](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/story_eda_parallel.png)
    *   Premissas QQPlot: [story_premissas_qqplot.png](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/story_premissas_qqplot.png)
    *   Premissas Resíduos: [story_premissas_residuals_fitted.png](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/story_premissas_residuals_fitted.png)
    *   Premissas ACF: [story_premissas_acf.png](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/story_premissas_acf.png)
    *   Premissas VIF: [story_premissas_vif.png](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/story_premissas_vif.png)
    *   Desempenho ROC: [story_desempenho_roc.png](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/story_desempenho_roc.png)
    *   Desempenho PR: [story_desempenho_pr.png](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/story_desempenho_pr.png)
    *   Desempenho Calibration: [story_desempenho_calibration.png](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/story_desempenho_calibration.png)
    *   Desempenho Lift & Gain: [story_desempenho_lift_gain.png](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/story_desempenho_lift_gain.png)
    *   Desempenho t-SNE: [story_desempenho_tsne.png](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/story_desempenho_tsne.png)
    *   Desempenho SHAP XGBoost: [story_desempenho_shap.png](file:///C:/Users/samue/.gemini/antigravity/brain/ceea9544-b160-4a30-9d87-8e0226e648e7/story_desempenho_shap.png)
