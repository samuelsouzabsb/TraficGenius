# Histórico de Conversa - Sessão 19c735cc

Este documento registra os diálogos, as decisões de design e as implementações realizadas na sessão atual de desenvolvimento do projeto **TraficGenius**.

---

## 1. Solicitações do Usuário e Contexto
* **Demanda Principal:** Integração de dois novos classificadores no pipeline de modelagem preditiva avançada: o 3º modelo (**Random Forest**) e o 4º modelo (**Regressão Logística**).
* **Demanda Visual:** Elaboração de um gráfico comparativo de performance com identidade visual neon escura (cores ciano e rosa neon) confrontando todos os modelos ativos com base nas métricas de Acurácia Global e F1-Score Macro.
* **Orientação a Testes (TDD):** Criação de testes unitários para a fase de modelagem e validação lógica.

---

## 2. Decisões de Design e Arquitetura

### 2.1. Escolha dos Hiperparâmetros
* **Random Forest Classifier:**
  * Instanciado com `n_estimators=100`, `max_depth=6`, `random_state=42` e `n_jobs=-1`.
  * *Raciocínio:* Limitar a profundidade máxima (`max_depth=6`) evita overfitting e reduz consideravelmente o tempo de execução local na máquina do usuário.
* **Logistic Regression:**
  * Instanciado com `max_iter=200`, `solver='lbfgs'`, `random_state=42` e `n_jobs=-1`.
  * *Raciocínio:* Modelo linear e simples que serve como uma excelente baseline clássica e interpretável.

### 2.2. Identidade Visual do Gráfico Comparativo (`comparativo_performance_modelos.png`)
* **Fundo (Background):** `#121214` (preto neon escuro).
* **Painel (Axes Facecolor):** `#1a1a1e`.
* **Grades (Grids):** `#2d2d34`.
* **Barras de F1-Score Macro:** `#ff007f` (rosa neon).
* **Barras de Acurácia Global:** `#00f0ff` (ciano neon).

---

## 3. Implementação e Resultados

### 3.1. Pipeline de Modelagem (`pipeline_fase3a5_modelagem.py`)
Atualizamos a rotina `run_pipeline` para instanciar, treinar e avaliar sequencialmente todos os classificadores:
1. **XGBoost (Tuned):** Acurácia: **71.89%** | F1-Score Macro: **43.10%** | ROC-AUC: **85.07%**
2. **CNN 1D:** Pulado automaticamente devido à falta da DLL do TensorFlow nativa (comportamento seguro/protegido).
3. **Random Forest:** Acurácia: **45.27%** | F1-Score Macro: **31.70%** | ROC-AUC: **81.89%**
4. **Regressão Logística:** Acurácia: **32.03%** | F1-Score Macro: **22.97%** | ROC-AUC: **69.78%**

O gráfico comparativo foi gerado e salvo em `dataset/comparativo_performance_modelos.png`.

### 3.2. Gerador de Notebook (`generate_notebook.py`)
Adicionamos as alterações para documentar o treinamento comparativo de 4 modelos na Fase 3-5 e carregar a visualização gráfica resultante no `notebook.ipynb`.

### 3.3. Testes Unitários (`tests/test_pipeline_modelagem.py`)
Desenvolvemos um arquivo de teste unitário validando as rotinas auxiliares de modelagem:
* Validação estatística de `get_chance_criteria` (cálculo de C.Max e C.Prop).
* Validação da criação da imagem neon por `plot_model_comparison` em diretório temporário.

---

## 4. Dicas de Inglês (English Tips)
1. **Model Comparison:** Comparação de modelos. Em contextos de machine learning, costuma-se criar um *leaderboard* (quadro de líderes/ranking) para ordenar a performance deles.
2. **Tuned Classifier:** Classificador sintonizado/otimizado. Refere-se a modelos cujos hiperparâmetros foram ajustados usando técnicas de busca como grid search ou random search.
3. **Baseline:** Linha de base. Trata-se do modelo mais simples ou do critério ao acaso (como o *chance criteria*) contra o qual modelos complexos são comparados para provar seu valor estatístico.
4. **Overfitting vs Underfitting:** *Overfitting* é o sobreajuste (quando o modelo decora o treino e falha no teste), enquanto *underfitting* é o subajuste (quando o modelo é simples demais e não aprende as regras nem no treino).
