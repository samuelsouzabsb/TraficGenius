# Relatório de Avaliação Comparativa de Novos Treinamentos

Este relatório resume os resultados de desempenho obtidos ao treinar o modelo de classificação de severidade de acidentes (**Leve/Moderado** [0] vs. **Grave/Fatal** [1]) sob diferentes configurações:

1. **Baseline**: Todo o histórico unificado (US + BR), sem flag de país.
2. **Temporal (2020+)**: Apenas acidentes de 2020 em diante (US + BR).
3. **EUA (Separado)**: Modelo treinado exclusivamente com dados dos EUA.
4. **Brasil (Separado)**: Modelo treinado exclusivamente com dados do Brasil.
5. **Flag de País**: Todo o histórico unificado, utilizando uma flag de país (`pais_US`) como variável preditora.

---

## 📊 Desempenho do Stacking Ensemble por Configuração

O Stacking Ensemble combina as previsões dos 5 classificadores base (Regressão Logística, LDA, Random Forest, Rede Neural MLP e XGBoost) através de um meta-aprendedor Logit.

| Configuração | Limiar Ótimo | Acurácia Global (Ótimo) | Acurácia Balanceada | F1-Score Classe Grave (1) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 0.27 | 72.88% | 70.18% | 50.36% | 78.26% |
| **Temporal (2020+)** | 0.20 | 76.50% | 68.32% | 39.40% | 76.71% |
| **EUA (Separado)** | 0.30 | 75.95% | 71.82% | 52.53% | 80.19% |
| **Brasil (Separado)** | 0.21 | 47.07% | 56.93% | 41.87% | 61.25% |
| **Flag de País** | 0.27 | 72.87% | 70.61% | 50.80% | 78.66% |

---

## 🏆 Melhor Algoritmo Individual por Configuração

Além do Stacking Ensemble, esta tabela mostra o algoritmo individual de melhor desempenho (baseado no F1-Score da Classe Grave) em cada treino.

| Configuração | Algoritmo Vencedor | Limiar Ótimo | Acurácia Global (Ótimo) | F1-Score Classe Grave (1) | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline** | Stacking Ensemble | 0.27 | 72.88% | 50.36% | 78.26% |
| **Temporal (2020+)** | Stacking Ensemble | 0.20 | 76.50% | 39.40% | 76.71% |
| **EUA (Separado)** | Stacking Ensemble | 0.30 | 75.95% | 52.53% | 80.19% |
| **Brasil (Separado)** | Stacking Ensemble | 0.21 | 47.07% | 41.87% | 61.25% |
| **Flag de País** | Stacking Ensemble | 0.27 | 72.87% | 50.80% | 78.66% |

---

## 🔍 Principais Conclusões Estatísticas

1. **Segmentação Geográfica (EUA vs. Brasil)**:
   - Os modelos específicos revelam se a dinâmica de acidentes é distinta. O modelo brasileiro costuma sofrer com menor volume de dados, mas pode revelar fatores locais diferentes dos EUA (como o impacto de postos de combustível, velocidade média ou infraestrutura rodoviária).
   
2. **Impacto da Flag de País**:
   - A inclusão da flag `pais_US` no modelo unificado serve como um calibrador de viés geográfico. Se a flag tiver alta importância no XGBoost, ela indica uma diferença estrutural significativa na proporção de acidentes graves entre os países.

3. **Recorte Temporal (Desde 2020)**:
   - Avalia se o comportamento dos acidentes mudou significativamente pós-pandemia ou com as novas configurações de tráfego, servindo como teste de conceito de adaptação temporal.

O gráfico de comparação geral foi gerado em `dataset/comparativo_geral_configuracoes.png`.
