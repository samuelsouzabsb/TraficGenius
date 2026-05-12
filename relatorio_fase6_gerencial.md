# Relatório Gerencial: Fatores de Risco e Severidade de Acidentes (Fase 6)

Com base na modelagem multivariada desenvolvida nos scripts anteriores (Logit Ordinal, Random Forest, MDA e Redes Neurais), isolamos estatisticamente os principais fatores preditivos que levam um acidente a "pular" de uma severidade menor (ex: 1 ou 2) para graus críticos (3 e 4).

> [!TIP]
> A interpretação abaixo é focada em métricas acionáveis para Gestores de Trânsito, baseando-se nos **Odds Ratios** do modelo Logit e no **Feature Importance** do Random Forest.

## 1. Condições Meteorológicas (Risco Natural)

O modelo estatístico comprovou que variáveis atmosféricas são os maiores catalisadores de severidade extrema:

* **Visibilidade (Visibility):** Possui relação inversa fortíssima com a severidade. O *Odds Ratio* do Logit indica que para cada **1 milha a menos de visibilidade**, a chance do acidente ser de Severidade 3 ou 4 aumenta exponencialmente. Neblina ou chuva densa são os maiores red flags.
* **Precipitação (Precipitation):** O Random Forest apontou que volumes de chuva anormais multiplicam o risco basal de um acidente grave, sobretudo quando combinados com quedas bruscas de temperatura (risco de gelo na pista).
* **Pressão Atmosférica:** Quedas súbitas de pressão atmosférica (indicativo de tempestades se formando) mostraram carga discriminante significativa na Análise Discriminante (MDA).

## 2. Infraestrutura e Pontos de Atenção (POIs)

Surpreendentemente, a presença de infraestrutura urbana reduz a chance de fatalidades extremas:

* **Cruzamentos e Semáforos (Crossing / Traffic_Signal):** O Logit Ordinal gerou coeficientes **negativos** (Odds Ratio < 1) para essas variáveis. Isso significa que, embora acidentes *ocorram* com muita frequência nesses locais, a probabilidade deles serem Grau 3 ou 4 é **reduzida**. Isso ocorre pela redução natural de velocidade dos motoristas nesses pontos.
* **Junções de Rodovias (Junction):** Ao contrário dos cruzamentos urbanos, acidentes perto de ramais/junções rodoviárias têm alta probabilidade de serem Grau 4, de acordo com as Redes Neurais e o Random Forest. A alta velocidade associada à mudança de faixa torna esses pontos críticos.

## 3. Dinâmica Temporal e Comportamental

* **Horário (Sunrise_Sunset):** Acidentes noturnos (`Night`) apresentam um salto probabilístico gigantesco para os níveis 3 e 4. O modelo paramétrico capturou uma forte interação entre baixa iluminação e as outras variáveis meteorológicas.

## 🚀 Recomendações Estratégicas para o Gestor

1. **Alerta Antecipado (Early Warning System):** A integração das previsões de visibilidade e temperatura do modelo de Machine Learning (como a Rede Neural desenvolvida) deve disparar alertas automáticos aos painéis eletrônicos nas rodovias.
2. **Foco Preventivo em Junções:** Como cruzamentos urbanos tendem a gerar acidentes de Severidade 1 e 2, o policiamento focado na mitigação de **mortalidade** (Severidade 4) deve focar patrulhamento nas junções de alto fluxo (Junctions) em condições de visibilidade inferior a 5 milhas.
3. **Campanhas Noturnas:** A variável noite/dia superou condições geográficas em peso discriminante. Redutores de velocidade interativos durante a noite reduzem significativamente o risco capturado pela estatística.

---
*Análise realizada pelo Cientista de Dados Sênior via pipeline multivariado estruturado em Python.*
