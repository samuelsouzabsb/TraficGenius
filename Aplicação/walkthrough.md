# Walkthrough - Análise da Influência da Clusterização Espacial

Este documento resume as tarefas realizadas e a validação experimental conduzida para responder à solicitação `/goal`.

## O que foi realizado
1. **Desenvolvimento do Script de Experimento:** Criado o arquivo [analise_influencia_clusterizacao.py](file:///C:/Users/samuelbarroso/Documents/Desenvolvimento/TraficGenius/analise_influencia_clusterizacao.py) para automatizar a extração estatística e a avaliação preditiva cruzada.
2. **Execução e Análise Estatística:** O script rodou com sucesso sobre uma amostra de 200.000 registros, calculando severidade, clima e infraestrutura por cluster.
3. **Treinamento e Avaliação Cruzada:** O modelo XGBoost foi avaliado em 4 setups diferentes para medir o impacto direto dos recursos de geolocalização.
4. **Geração de Gráficos:** Salvos três gráficos elucidativos na pasta do projeto.
5. **Relatório da Análise:** Compilado relatório completo em formato Markdown no arquivo [relatorio_analise_clusterizacao.md](file:///C:/Users/samuelbarroso/Documents/Desenvolvimento/TraficGenius/relatorio_analise_clusterizacao.md).

## Resultados e Validação
- Os gráficos mostram claramente o impacto geográfico: o **Cluster 11** concentra o maior índice de letalidade (10.02% de óbitos), enquanto o **Cluster 13** concentra a maior densidade urbana (32% semáforos, 33% cruzamentos).
- A acurácia do XGBoost decaiu de **52.10%** para **45.10%** sem coordenadas ou clusters espaciais, comprovando a eficácia das features espaciais na modelagem de riscos viários.
