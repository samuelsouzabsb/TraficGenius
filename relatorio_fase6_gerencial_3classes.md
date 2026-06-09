# Relatório Gerencial: Dinâmica e Explicabilidade de Acidentes (Fase 6 - 3 Classes)

Com base na migração metodológica para **3 classes de severidade** (Leve/Médio, Grave e Fatal) e o treinamento dos novos classificadores, conseguimos isolar com muito mais estabilidade preditiva os fatores que desencadeiam acidentes graves e fatais.

> [!NOTE]
> Ao agrupar os acidentes de menor impacto (Graus 1 e 2 originais) na categoria unificada **Leve/Médio**, reduzimos o ruído de fronteira decisória do modelo. A acurácia global do nosso classificador XGBoost subiu para **57.91%** e a área sob a curva ROC-AUC Macro alcançou **84.51%**.

---

## 1. Zonas de Risco Geográfico (Clustering Espacial)

A clusterização espacial *K-Means* mostrou forte influência na previsão de severidade mesmo sob a nova ótica de 3 classes:
* **Predisposição Física das Rodovias:** Determinados clusters (como o Cluster 3 e o Cluster 5) concentram rodovias intermunicipais de pista simples e fluxo rápido. Nesses locais, o modelo de 3 classes atribui maior probabilidade de ocorrências **Graves** ou **Fatais** devido ao risco de colisões frontais em alta velocidade, independentemente do horário ou do clima favorável.

## 2. Visibilidade e Precipitação (Análise SHAP)

O SHAP (*SHapley Additive exPlanations*) quantificou o impacto das variáveis climáticas com maior clareza:
* **Fator Visibilidade:** A redução da visibilidade para menos de 3 milhas atua como o principal gatilho que desloca a predição da categoria *Leve/Médio* diretamente para *Grave*.
* **Efeito Acumulativo de Precipitação:** O XGBoost e o Random Forest de 3 classes capturaram interações não lineares onde volumes de chuva superiores a 0.05 polegadas agem como multiplicadores de severidade, em especial quando a pista tem histórico de baixa aderência (distância de frenagem comprometida).

## 3. Dinâmica Temporal: Perigos Noturnos

* **O Efeito da Madrugada:** O modelo identificou que acidentes ocorridos de madrugada (entre 2h e 5h) têm uma taxa de severidade **Fatal** significativamente maior. 
* **Horários de Pico:** O volume de colisões é maior nos picos (7h-9h e 16h-18h), mas o modelo prevê com alta probabilidade que estes acidentes serão de categoria **Leve/Médio**, pois o congestionamento urbano reduz a velocidade média das colisões (choques traseiros simples).

---

## 🚀 Recomendações Estratégicas para Gestão de Tráfego

1. **Sinalização Climática Inteligente nos Clusters Críticos:** Instalar painéis de mensagem variável (PMVs) nas rodovias dos Clusters 3 e 5. A velocidade limite das vias deve ser reduzida dinamicamente via software sempre que sensores locais indicarem visibilidade abaixo de 3.0 milhas ou precipitação contínua.
2. **Priorização de Equipes de Socorro Avançado:** Posicionar UTIs móveis e patrulhas de emergência em pontos de apoio nas rodovias dos clusters de risco durante a madrugada (2h às 5h), focando em tempos de resposta extremamente curtos para acidentes classificados como potencialmente Graves ou Fatais.
3. **Integração com APIs de Clima para Prevenção Ativa:** Utilizar o novo endpoint `/api/predict-3classes/` integrado com previsões de radares climáticos para disparar alertas preditivos automáticos às concessionárias de rodovias sobre o aumento de risco nas próximas horas.

---
*Análise estruturada e refinada com base no pipeline otimizado de Machine Learning e Explainable AI (3 Classes).*
