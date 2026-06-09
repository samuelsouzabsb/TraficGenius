# Walkthrough - Implementação do Pipeline e Painel de 3 Classes (TraficGenius)

Este guia explica detalhadamente a arquitetura de 3 classes implementada no projeto **TraficGenius** de forma paralela e independente à versão original de 4 classes.

## 1. O que foi modificado e criado?

Para manter os arquivos originais de 4 classes totalmente funcionais e intocados, todas as novas implementações foram salvas em arquivos novos e paralelos com o sufixo `_3classes`.

```
TraficGenius/
│
├── dataset/
│   ├── xgboost_model_3classes.joblib          # Modelo XGBoost de 3 classes
│   ├── random_forest_model_3classes.joblib    # Modelo Random Forest de 3 classes
│   ├── logistic_regression_model_3classes.joblib # Modelo Regr. Logística de 3 classes
│   ├── scaler_3classes.joblib                 # StandardScaler de 3 classes
│   └── frontend/
│       ├── dashboard_data_3classes.json       # Dados de KPI estáticos de 3 classes
│       └── history/
│           └── historical_data_3classes.json  # Histórico estático de 3 classes
│
├── backend/api/
│   ├── views.py                               # Adaptado para carregar modelos 3C e tratar '?classes=3'
│   └── tests.py                               # Suíte de testes expandida para 10 testes (TDD)
│
├── frontend/
│   ├── index_3classes.html                    # Dashboard HTML específico de 3 classes
│   ├── app_3classes.js                        # Lógica e requisições HTTP específicas para 3 classes
│   ├── charts_3classes.js                     # Configuração Chart.js de 3 classes (Ciano, Laranja, Vermelho)
│   ├── map_3classes.js                        # Marcadores Leaflet específicos para 3 classes
│   └── modelos_3classes.html                  # Comparativo estatístico e simulação em tempo real
│
├── pipeline_fase3a5_modelagem_3classes.py     # Pipeline de modelagem de 3 classes
├── train_all_models_3classes.py               # Script para treinar e salvar os modelos de 3 classes
├── avaliar_modelos_detalhado_3classes.py       # Avaliador estatístico dos modelos 3C
├── relatorio_metricas_detalhadas_3classes.md   # Relatório com Matriz de Confusão, F1 e ROC-AUC
└── walkthrough_3classes.md                    # Este guia
```

---

## 2. Nova Escala de Severidade (3 Classes)

A severidade foi mapeada no pipeline seguindo as regras de trânsito sugeridas:
- **Classe 1 (Leve/Médio):** Agrupa os graus 1 e 2 originais (Representados pela cor **Ciano**).
- **Classe 2 (Grave):** Corresponde ao grau 3 original (Representado pela cor **Laranja**).
- **Classe 3 (Fatal):** Corresponde ao grau 4 original (Representado pela cor **Vermelho/Rosa**).

Com a simplificação do problema para 3 classes, a acurácia global do classificador **XGBoost** subiu de **51.81%** (4 classes) para **57.91%** (3 classes), representando uma melhoria absoluta de **+6.1%** no poder preditivo geral!

---

## 3. Endpoints da API REST (Backend)

O backend Django foi modificado para suportar dinamicamente os dois formatos de classificação:
1. **`/api/acidentes/?classes=3`**: Retorna os acidentes históricos com a severidade convertida dinamicamente para a escala de 1 a 3. Permite filtragem por severidade compatível (`?severidade=1` retornará acidentes originais 1 e 2).
2. **`/api/acidentes/<id>/?classes=3`**: Retorna os detalhes do acidente específico com a severidade mapeada e calcula as predições de todos os modelos em tempo real utilizando os novos arquivos `.joblib` de 3 classes.
3. **`/api/dashboard-stats-3classes/`**: Retorna os KPIs, o histograma de distribuição temporal e o SHAP agrupado especificamente para a versão de 3 classes.
4. **`/api/predict-3classes/`**: Permite predições manuais em tempo real enviando as 26 variáveis e retornando as probabilidades de Leve/Médio, Grave e Fatal.

---

## 4. Como rodar e testar localmente?

### Passo 1: Executar Testes Automatizados (Backend)
Para validar que nenhuma funcionalidade foi quebrada e que os testes de 3 classes estão passando:
```powershell
python backend/manage.py test api
```
Você verá o output: **`Ran 10 tests in ...s - OK`**.

### Passo 2: Iniciar o Servidor Django
Se certifique de que o servidor local está de pé na porta 8000:
```powershell
python backend/manage.py runserver
```

### Passo 3: Abrir a Interface no Navegador
- Para acessar a versão de **4 Classes** original: Abra o arquivo `frontend/index.html` ou acesse seu servidor HTTP local normal.
- Para acessar a versão de **3 Classes** nova: Abra o arquivo `frontend/index_3classes.html` no seu navegador.
- Para navegar para a página de arquiteturas de 3 classes: Clique em **Models** no cabeçalho ou abra o arquivo `frontend/modelos_3classes.html`.

---

## 5. Resultados de Desempenho (3 Classes)

### XGBoost (Primary Champion)
- **Acurácia Global:** 57.91%
- **Macro ROC-AUC:** 84.51%
- **F1-Score por Classe:**
  - Leve/Médio: **68.23%** (Recall: 53.06%)
  - Grave: **53.56%** (Recall: 77.58%)
  - Fatal: **16.30%** (Recall: 80.01%)

### Random Forest
- **Acurácia Global:** 50.19%
- **Macro ROC-AUC:** 79.65%

### Regressão Logística
- **Acurácia Global:** 43.03%
- **Macro ROC-AUC:** 67.39%
