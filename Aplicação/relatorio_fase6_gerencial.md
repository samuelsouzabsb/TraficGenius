# Relatório Gerencial: Dinâmica e Explicabilidade de Acidentes Severos (Fase 6)

Com base no novo pipeline avançado de Machine Learning, que agora integra **Gradient Boosting (XGBoost)**, **Redes Neurais Convolucionais (CNN 1D)** e Explicabilidade de IA através de **Valores SHAP (Teoria dos Jogos)**, conseguimos isolar com precisão cirúrgica os maiores causadores de acidentes fatais ou gravíssimos (Severidade 3 e 4).

> [!TIP]
> Esta versão do relatório foca na interpretabilidade (*Explainable AI*). Os Valores SHAP nos mostram não apenas *quais* variáveis importam, mas *como* elas empurram a probabilidade para um acidente letal.

## 1. Zonas de Risco e Clustering (Engenharia Espacial)

A nossa clusterização não supervisionada (*K-Means*) confirmou a hipótese de que a geografia importa tanto quanto o clima.
* **Clusters de Alta Severidade:** Certas rodovias intermunicipais, agora agrupadas nos nossos "Clusters Espaciais", apresentaram uma predisposição gigantesca para acidentes Grau 4, independente do clima. O modelo aprendeu que essas vias, por permitirem alta velocidade sem fiscalização adequada, são letais.

## 2. A Força das Variáveis Meteorológicas (Análise SHAP)

Ao rodar o SHAP sobre o XGBoost, o impacto do clima foi quantificado com exatidão:
* **Visibilidade (Visibility):** É o fator número 1. O gráfico SHAP mostra um "ponto de virada" claro: quando a visibilidade cai abaixo de 2 milhas, o risco de Severidade 4 dispara.
* **Precipitação (Precipitation):** Diferente de chuvas leves, precipitações acima de um determinado limiar mostraram um efeito multiplicador no risco (interação não linear capturada fortemente pela CNN 1D).

## 3. Dinâmica Temporal: O Efeito da Madrugada

* **Hora do Dia:** A nova feature extraída revelou que acidentes ocorridos de madrugada (entre 2h e 5h da manhã) possuem a maior probabilidade de letalidade. Durante os horários de pico (7h-9h e 16h-18h), há **mais acidentes em volume**, mas a imensa maioria é de **Severidade 2** (batidas leves devido ao trânsito lento).

## 🚀 Recomendações Estratégicas para o Gestor (Data-Driven)

1. **Gestão de Tráfego Dinâmica:** Implementar redutores de velocidade eletrônicos ativados por sensores climáticos sempre que a visibilidade cair para perto de 2 milhas.
2. **Realocação de Patrulhamento:** Transferir o efetivo de resgate pesado (ambulâncias avançadas) para os *Clusters Espaciais* identificados pelo algoritmo (zonas rurais/rodoviárias) durante o período da madrugada. Acidentes em centros urbanos no horário de pico requerem guinchos rápidos, não UTIs móveis.
3. **Mapeamento Preditivo:** Alimentar a Rede Neural CNN com previsões meteorológicas em tempo real para antecipar em horas onde os recursos de emergência serão mais requisitados, alocando-os preventivamente.

---
*Análise realizada pelo Cientista de Dados Sênior via pipeline multivariado estruturado em Python, utilizando modelos de Estado da Arte (XAI, XGBoost, Deep Learning).*
