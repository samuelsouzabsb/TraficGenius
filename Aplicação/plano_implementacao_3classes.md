# Plano de Implementação: Transição Suave para 3 Classes no TraficGenius

Este plano detalha as alterações necessárias para criar uma versão paralela completa do **TraficGenius** com **3 classes de severidade** (Leve/Médio, Grave e Fatal) em vez das 4 originais, preservando todos os arquivos da versão de 4 classes intactos.

## Objetivos e Mapeamento de Classes
No novo modelo, a severidade original do banco de dados (1 a 4) será mapeada da seguinte forma:
- **Severidade 1 (Leve) e 2 (Moderada):** Mapeadas para **Classe 1 (Leve/Médio)**.
- **Severidade 3 (Grave):** Mapeada para **Classe 2 (Grave)**.
- **Severidade 4 (Fatal):** Mapeada para **Classe 3 (Fatal)**.

---

## Proposta de Novos Arquivos (Versão de 3 Classes)

### 1. Pipelines de Modelagem e Treinamento (Fases 3 a 5)
Para treinar e persistir os modelos de 3 classes isoladamente:

#### [NEW] [pipeline_fase3a5_modelagem_3classes.py](pipeline_fase3a5_modelagem_3classes.py)
* Pipeline completo que mapeia o target para 3 classes, calcula os pesos balanceados de classes e treina o XGBoost, gerando `dataset/xgboost_model_3classes.joblib` e `dataset/scaler_3classes.joblib`.

#### [NEW] [train_all_models_3classes.py](train_all_models_3classes.py)
* Script para treinar e salvar os classificadores XGBoost, Random Forest e Regressão Logística específicos de 3 classes no formato `.joblib`.

#### [NEW] [avaliar_modelos_detalhado_3classes.py](avaliar_modelos_detalhado_3classes.py)
* Avaliará os 3 modelos de forma isolada, gerando matrizes de confusão de 3x3 e métricas de F1, Recall e ROC-AUC (OVR Macro/Weighted), gravando o resultado no novo relatório [relatorio_metricas_detalhadas_3classes.md](relatorio_metricas_detalhadas_3classes.md).

---

### 2. Exportadores de Dados do Frontend
Para alimentar as estatísticas do novo painel estático:

#### [NEW] [export_frontend_data_3classes.py](export_frontend_data_3classes.py)
* Consolida os KPIs e as séries temporais divididos em Leve/Médio (G1) vs Grave/Fatal (G2 e G3) e exporta para `frontend/dashboard_data_3classes.json`.

#### [NEW] [export_historical_data_3classes.py](export_historical_data_3classes.py)
* Agrega e exporta a distribuição do donut de severidade (3 classes) e radar de infraestrutura para `frontend/history/historical_data_3classes.json`.

---

### 3. Backend e APIs (Django)
Para suportar o SaaS em tempo real e consultas dinâmicas de 3 classes:

#### [MODIFY] [views.py](backend/api/views.py)
* Adicionar lógica lazy-loading para os novos modelos de 3 classes.
* Criar a `PredictionView3Classes` no endpoint `/api/predict_3classes/` que realiza predições e retorna a distribuição de probabilidades das 3 novas categorias.
* Criar a `DashboardStatsView3Classes` no endpoint `/api/stats_3classes/` para prover as agregação do banco de dados mapeadas dinamicamente para as 3 classes de severidade.

#### [MODIFY] [urls.py](backend/api/urls.py)
* Registrar as novas rotas de predição e estatísticas: `/api/predict_3classes/` e `/api/stats_3classes/`.

#### [MODIFY] [tests.py](backend/api/tests.py)
* Implementar testes automatizados cobrindo os novos endpoints de 3 classes (validação de payloads, respostas de probabilidade tridimensionais, tratamentos de erro e segurança).

---

### 4. Painel de Visualização Frontend (3 Classes)
Para permitir que o usuário interaja visualmente com o novo formato:

#### [NEW] [index_3classes.html](frontend/index_3classes.html)
* Estrutura HTML do painel adaptada para as legendas, cores e KPIs específicos das 3 classes.

#### [NEW] [app_3classes.js](frontend/app_3classes.js)
* Orquestrador assíncrono que carrega `dashboard_data_3classes.json` ou consome a API `/api/stats_3classes/` e `/api/predict_3classes/`.

#### [NEW] [charts_3classes.js](frontend/charts_3classes.js)
* Configurações de cores de gráficos do Chart.js específicas de 3 classes (Donut com 3 categorias: Leve/Médio em ciano, Grave em rosa e Fatal em vermelho).

#### [NEW] [map_3classes.js](frontend/map_3classes.js)
* Lógica do mapa Leaflet para plotagem de marcadores utilizando a nova escala de gravidade tricolor.

#### [NEW] [modelos_3classes.html](frontend/modelos_3classes.html)
* Página de visão geral de modelos interativa que consulta o endpoint `/api/predict_3classes/` e exibe as simulações em tempo real.

---

## Plano de Verificação

### Testes Automatizados (TDD)
- Executar os novos testes em Django: `python backend/manage.py test backend.api.tests`
- Executar testes gerais de regressão: `python -m unittest discover -s tests`

### Verificação Manual
- Treinar os modelos de 3 classes executando `train_all_models_3classes.py`.
- Gerar os relatórios detalhados com `avaliar_modelos_detalhado_3classes.py`.
- Iniciar o servidor Django (`python backend/manage.py runserver`) e acessar o endpoint `/api/stats_3classes/` e `/api/predict_3classes/` pelo Postman ou navegador para verificar integridade e segurança.
- Abrir [index_3classes.html](frontend/index_3classes.html) e verificar o funcionamento correto dos gráficos de 3 classes e da plotagem do mapa.
