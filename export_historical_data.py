# -*- coding: utf-8 -*-
"""
Exportador de Dados Históricos (Historical Data Exporter)
Este script agrega e exporta dados consolidados históricos a partir do parquet higienizado,
gerando estatísticas avançadas estruturadas para múltiplos gráficos do painel secundário (V3).
Exporta informações como distribuição temporal dia/noite, condições climáticas multivariadas,
ocorrência de sinalizações de trânsito e infraestruturas, e distância média do trecho afetado.

Dicas de Inglês (English Tips):
- 'Historical data' significa dados históricos.
- 'Radar chart' é um gráfico de radar/teia de aranha (usado para comparar múltiplas variáveis quantitativas).
- 'Donut chart' é o gráfico de rosca (semelhante ao gráfico de pizza/pie chart, mas com furo central).
- 'Histogram' é o histograma (gráfico de distribuição de frequências).
- 'Subset' significa subconjunto (verificação se um grupo de colunas existe no DataFrame).
"""

import pandas as pd
import numpy as np
import json
import os

def export_historical_data():
    """
    Agrega as informações estatísticas complexas e históricas a partir do arquivo Parquet.
    Formata o JSON e salva na pasta dedicada de histórico do frontend.
    """
    print("Exportando dados historicos para o Frontend (V3)...")
    
    dataset_path = r"c:\Users\samuelbarroso\Documents\Desenvolvimento\TraficGenius\dataset\dataset_amostra_limpa_avancado.parquet"
    
    # Estrutura inicial contendo os dados mockados/fallback e formatos de chaves esperados
    data = {
        "time_vs_sun": { "labels": [f"{i}h" for i in range(24)], "day": [], "night": [] },
        "weather_matrix": [],  # Matriz de espalhamento multivariada (Scatter plot data)
        "infra_radar": { "labels": ["Cruzamento", "Juncao", "Semaforo", "Estacao", "Pare"], "values": [] },
        "severity_donut": { "labels": ["G1", "G2", "G3", "G4"], "values": [] },
        "distance_hist": { "labels": ["< 1mi", "1-3mi", "3-5mi", "> 5mi"], "values": [] }
    }
    
    # Verifica a existência do arquivo Parquet limpo para iniciar agregação real
    if os.path.exists(dataset_path):
        df = pd.read_parquet(dataset_path)
        
        # 1. Agregação Temporal Dividida por Período de Sol (Time vs Sun)
        # Calcula a frequência horária de incidentes segregando se ocorreram durante o Dia ou Noite
        if 'Hora_do_Dia' in df.columns and 'Nascer_Por_Sol' in df.columns:
            for h in range(24):
                hour_data = df[df['Hora_do_Dia'] == h]
                day = len(hour_data[hour_data['Nascer_Por_Sol'] == 'Day'])
                night = len(hour_data[hour_data['Nascer_Por_Sol'] == 'Night'])
                data["time_vs_sun"]["day"].append(day)
                data["time_vs_sun"]["night"].append(night)
        
        # 2. Matriz Climática Multivariada (Weather Matrix - Scatter Sample)
        # Seleciona uma amostra aleatória de 300 pontos contendo temperatura, visibilidade, umidade e severidade.
        # Utilizado para mapeamento tridimensional ou correlação visual.
        if set(['Temperatura_F', 'Visibilidade_Milhas', 'Umidade_Percentual', 'Severidade']).issubset(df.columns):
            weather_sample = df.dropna(subset=['Temperatura_F', 'Visibilidade_Milhas', 'Umidade_Percentual']).sample(min(300, len(df)))
            for _, row in weather_sample.iterrows():
                data["weather_matrix"].append({
                    "temp": float(row['Temperatura_F']),
                    "vis": float(row['Visibilidade_Milhas']),
                    "hum": float(row['Umidade_Percentual']),
                    "sev": int(row['Severidade'])
                })
        
        # 3. Presença de Infraestrutura Rodoviária (Infrastructure Radar - Cruzamento, Juncao, Semaforo, etc.)
        # Soma a presença booleana (convertida para 0 ou 1) de elementos de via física no local dos acidentes.
        infra_cols = ["Cruzamento", "Juncao", "Semaforo", "Estacao", "Pare"]
        for col in infra_cols:
            if col in df.columns:
                count = int(df[col].sum())
                data["infra_radar"]["values"].append(count)
            else:
                data["infra_radar"]["values"].append(0)
                
        # 4. Distribuição Geral das Quatro Classes de Severidade (Severity Donut Chart)
        if 'Severidade' in df.columns:
            for s in [1, 2, 3, 4]:
                count = len(df[df['Severidade'] == s])
                data["severity_donut"]["values"].append(count)
                
        # 5. Histograma da Extensão de Congestionamento (Distance Hist)
        # Classifica a distância da rodovia impactada pelo acidente em faixas definidas (bins)
        if 'Distancia_Milhas' in df.columns:
            d = df['Distancia_Milhas']
            data["distance_hist"]["values"] = [
                len(d[d < 1]),
                len(d[(d >= 1) & (d < 3)]),
                len(d[(d >= 3) & (d < 5)]),
                len(d[d >= 5])
            ]
    else:
        # Se a base limpa não existir, carrega dados mockados consistentes com a escala do projeto (Fallback)
        print("Dataset avançado não encontrado. Exportando fallback data.")
        data["time_vs_sun"]["day"] = [np.random.randint(50, 150) for _ in range(24)]
        data["time_vs_sun"]["night"] = [np.random.randint(20, 100) for _ in range(24)]
        data["weather_matrix"] = [{"temp": np.random.uniform(20, 90), "vis": np.random.uniform(1, 10), "hum": np.random.uniform(30, 90), "sev": np.random.randint(1, 5)} for _ in range(100)]
        data["infra_radar"]["values"] = [1500, 2300, 3100, 400, 800]
        data["severity_donut"]["values"] = [5000, 15000, 3000, 800]
        data["distance_hist"]["values"] = [12000, 5000, 1500, 500]

    # Cria o subdiretório de histórico na pasta do frontend se ainda não existir
    history_dir = r"c:\Users\samuelbarroso\Documents\Desenvolvimento\TraficGenius\frontend\history"
    os.makedirs(history_dir, exist_ok=True)
    
    out_file = os.path.join(history_dir, 'historical_data.json')
    # Salva no disco codificado em UTF-8
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print(f"Dados históricos exportados para: {out_file}")

if __name__ == "__main__":
    export_historical_data()
