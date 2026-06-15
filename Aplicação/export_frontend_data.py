# -*- coding: utf-8 -*-
"""
Exportador de Dados do Dashboard (Frontend Data Exporter)
Este script processa o conjunto de dados higienizado e extrai indicadores-chave de desempenho (KPIs),
estatísticas de severidade ao longo do dia, dados de explicabilidade SHAP mockados e
coordenadas geográficas prontas para serem mapeadas no dashboard do frontend.

Dicas de Inglês (English Tips):
- 'KPIs' (Key Performance Indicators) significa Indicadores-Chave de Desempenho.
- 'Static data' significa dados estáticos (pré-calculados para leitura rápida).
- 'Mock/Mocked data' refere-se a dados simulados ou falsos, estruturados para fins de testes ou apresentação.
- 'Fallback' significa um plano de recuo (dados de reserva utilizados caso a base real de entrada falhe).
"""

import pandas as pd
import numpy as np
import json
import os

def export_data():
    """
    Agrega as informações estatísticas e geográficas do arquivo Parquet de amostra limpa.
    Formata tudo em uma única estrutura hierárquica e a exporta como um arquivo JSON de dados.
    """
    print("Exportando dados estáticos para o Frontend...")
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(project_root, "dataset", "dataset_amostra_limpa_avancado.parquet")
    
    # Estrutura JSON inicial padrão (com valores reais e mockados para fallback)
    data = {
        "kpis": {
            "avg_visibility": 0.0,
            "max_severity": 4,
            "accuracy": 0.707  # Acurácia de 70.7% obtida pelo XGBoost Tuned na base completa
        },
        "shap_data": {
            "labels": ["Visibilidade_Milhas", "Hora_do_Dia", "Temperatura_F", "Precipitacao_Polegadas", "Velocidade_Vento_Mph", "Umidade_Percentual"],
            "values": [2.8, 2.1, 1.5, 1.1, 0.8, 0.4]  # Média absoluta simulada dos valores SHAP (impacto das variáveis)
        },
        "time_data": {
            "labels": [f"{i}h" for i in range(24)],  # Rótulos para as 24 horas do dia (0h às 23h)
            "low_severity": [],
            "high_severity": []
        },
        "map_clusters": []  # Coordenadas geográficas dos acidentes
    }
    
    # 1. Tenta carregar e processar a base real limpa
    if os.path.exists(dataset_path):
        df = pd.read_parquet(dataset_path)
        
        # 1.1 Calcula a média da variável de Visibilidade e KPIs associados
        data["kpis"]["avg_visibility"] = float(df['Visibilidade_Milhas'].mean())
        data["kpis"]["max_severity"] = int(df['Severidade'].max())
        data["kpis"]["accuracy"] = 0.707
        
        # 1.2 Agrupa quantidade de acidentes por hora dividindo por gravidade:
        # Baixa Gravidade (Severidade 1 e 2) vs Alta Gravidade (Severidade 3 e 4)
        if 'Hora_do_Dia' in df.columns:
            for h in range(24):
                hour_data = df[df['Hora_do_Dia'] == h]
                low_sev = len(hour_data[hour_data['Severidade'].isin([1, 2])])
                high_sev = len(hour_data[hour_data['Severidade'].isin([3, 4])])
                
                data["time_data"]["low_severity"].append(low_sev)
                data["time_data"]["high_severity"].append(high_sev)
        else:
            # Fallback local de preenchimento randômico caso ocorra problema na coluna
            data["time_data"]["low_severity"] = [np.random.randint(50, 300) for _ in range(24)]
            data["time_data"]["high_severity"] = [np.random.randint(10, 100) for _ in range(24)]
            
        # 1.3 Coleta uma amostra de 100 pontos geográficos para plotar no mapa Leaflet.
        # Amostra 50 acidentes graves e 50 acidentes leves de forma balanceada para evitar sobrecarga no render
        df_severe = df[df['Severidade'].isin([3, 4])].sample(min(50, len(df[df['Severidade'].isin([3, 4])])))
        df_low = df[df['Severidade'].isin([1, 2])].sample(min(50, len(df[df['Severidade'].isin([1, 2])])))
        
        for _, row in pd.concat([df_severe, df_low]).iterrows():
            data["map_clusters"].append({
                "lat": float(row['Latitude_Inicial']),
                "lng": float(row['Longitude_Inicial']),
                "severity": int(row['Severidade'])
            })
    else:
        # 2. Caminho de Fallback caso o pipeline anterior não tenha sido executado
        print("Dataset avançado não encontrado. Exportando fallback data.")
        data["kpis"]["avg_visibility"] = 5.4
        data["time_data"]["low_severity"] = [np.random.randint(50, 300) for _ in range(24)]
        data["time_data"]["high_severity"] = [np.random.randint(10, 100) for _ in range(24)]
        
    # Define o diretório de saída na pasta do frontend
    frontend_dir = os.path.join(project_root, "frontend")
    os.makedirs(frontend_dir, exist_ok=True)
    
    out_file = os.path.join(frontend_dir, 'dashboard_data.json')
    # Grava os dados formatados em JSON usando codificação UTF-8 e recuo para leitura amigável
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print(f"Dados exportados com sucesso para: {out_file}")

if __name__ == "__main__":
    export_data()
