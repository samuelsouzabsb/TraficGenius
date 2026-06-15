# -*- coding: utf-8 -*-
"""
Exportador de Dados Históricos (3 Classes)
Gera estatísticas consolidadas históricas para o dashboard de 3 classes (Donut 3 classes).
"""

import pandas as pd
import numpy as np
import json
import os

def export_historical_data_3classes():
    print("Exportando dados históricos para o Frontend V3 (3 Classes)...")
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(project_root, "dataset", "dataset_amostra_limpa_avancado.parquet")
    
    data = {
        "time_vs_sun": { "labels": [f"{i}h" for i in range(24)], "day": [], "night": [] },
        "weather_matrix": [],
        "infra_radar": { "labels": ["Cruzamento", "Juncao", "Semaforo", "Estacao", "Pare"], "values": [] },
        "severity_donut": { "labels": ["Leve/Médio", "Grave", "Fatal"], "values": [] },
        "distance_hist": { "labels": ["< 1mi", "1-3mi", "3-5mi", "> 5mi"], "values": [] }
    }
    
    if os.path.exists(dataset_path):
        df = pd.read_parquet(dataset_path)
        
        # 1. Time vs Sun
        if 'Hora_do_Dia' in df.columns and 'Nascer_Por_Sol' in df.columns:
            for h in range(24):
                hour_data = df[df['Hora_do_Dia'] == h]
                day = len(hour_data[hour_data['Nascer_Por_Sol'] == 'Day'])
                night = len(hour_data[hour_data['Nascer_Por_Sol'] == 'Night'])
                data["time_vs_sun"]["day"].append(day)
                data["time_vs_sun"]["night"].append(night)
        
        # 2. Weather Matrix (mapeando severidade para 3 classes: 1,2->1, 3->2, 4->3)
        if set(['Temperatura_F', 'Visibilidade_Milhas', 'Umidade_Percentual', 'Severidade']).issubset(df.columns):
            weather_sample = df.dropna(subset=['Temperatura_F', 'Visibilidade_Milhas', 'Umidade_Percentual']).sample(min(300, len(df)))
            for _, row in weather_sample.iterrows():
                orig_sev = int(row['Severidade'])
                mapped_sev = 1 if orig_sev in [1, 2] else (2 if orig_sev == 3 else 3)
                data["weather_matrix"].append({
                    "temp": float(row['Temperatura_F']),
                    "vis": float(row['Visibilidade_Milhas']),
                    "hum": float(row['Umidade_Percentual']),
                    "sev": mapped_sev
                })
        
        # 3. Infra Radar
        infra_cols = ["Cruzamento", "Juncao", "Semaforo", "Estacao", "Pare"]
        for col in infra_cols:
            if col in df.columns:
                count = int(df[col].sum())
                data["infra_radar"]["values"].append(count)
            else:
                data["infra_radar"]["values"].append(0)
                
        # 4. Severity Donut (3 Classes)
        if 'Severidade' in df.columns:
            # Mapeia quantidades:
            # Leve/Médio = 1 e 2
            count_lm = len(df[df['Severidade'].isin([1, 2])])
            count_g = len(df[df['Severidade'] == 3])
            count_f = len(df[df['Severidade'] == 4])
            
            data["severity_donut"]["values"] = [count_lm, count_g, count_f]
                
        # 5. Distance Hist
        if 'Distancia_Milhas' in df.columns:
            d = df['Distancia_Milhas']
            data["distance_hist"]["values"] = [
                len(d[d < 1]),
                len(d[(d >= 1) & (d < 3)]),
                len(d[(d >= 3) & (d < 5)]),
                len(d[d >= 5])
            ]
    else:
        print("Dataset avançado não encontrado. Exportando fallback data.")
        data["time_vs_sun"]["day"] = [np.random.randint(50, 150) for _ in range(24)]
        data["time_vs_sun"]["night"] = [np.random.randint(20, 100) for _ in range(24)]
        data["weather_matrix"] = [{"temp": np.random.uniform(20, 90), "vis": np.random.uniform(1, 10), "hum": np.random.uniform(30, 90), "sev": np.random.randint(1, 4)} for _ in range(100)]
        data["infra_radar"]["values"] = [1500, 2300, 3100, 400, 800]
        data["severity_donut"]["values"] = [20000, 3000, 800]
        data["distance_hist"]["values"] = [12000, 5000, 1500, 500]
 
    history_dir = os.path.join(project_root, "frontend", "history")
    os.makedirs(history_dir, exist_ok=True)
    
    out_file = os.path.join(history_dir, 'historical_data_3classes.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print(f"Dados históricos 3 classes exportados para: {out_file}")

if __name__ == "__main__":
    export_historical_data_3classes()
