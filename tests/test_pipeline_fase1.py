# -*- coding: utf-8 -*-
"""
Testes Unitários do Pipeline Fase 1 (Pipeline Phase 1 Unit Tests)
Este script valida a lógica de renomeação de colunas e integridade de dados da Fase 1.

Dicas de Inglês (English Tips):
- 'Rename logic' = Lógica de renomeação.
- 'Assertion' = Asserção/validação lógica.
- 'Column alignment' = Alinhamento de colunas.
"""

import unittest
import pandas as pd
# Importa o dicionário e a função a serem testados
from pipeline_fase1_eda import rename_columns, COLUMNS_MAPPING

class TestPipelineFase1(unittest.TestCase):
    
    def setUp(self):
        """
        Prepara um DataFrame fictício com colunas originais em inglês.
        """
        # Cria dados dummy com nomes de colunas em inglês
        self.dummy_data = pd.DataFrame({
            'Severity': [2, 3],
            'Start_Time': ['2021-01-01 00:00:00', '2021-01-01 01:00:00'],
            'Start_Lat': [34.05, 40.71],
            'Start_Lng': [-118.24, -74.00],
            'Distance(mi)': [0.5, 1.2],
            'Temperature(F)': [70.5, 65.0],
            'Wind_Chill(F)': [70.5, 65.0],
            'Humidity(%)': [50.0, 60.0],
            'Pressure(in)': [29.9, 30.0],
            'Visibility(mi)': [10.0, 8.0],
            'Wind_Speed(mph)': [5.0, 10.0],
            'Precipitation(in)': [0.0, 0.1],
            'Amenity': [False, True],
            'Bump': [False, False],
            'Crossing': [True, False],
            'Give_Way': [False, False],
            'Junction': [False, True],
            'No_Exit': [False, False],
            'Railway': [False, False],
            'Roundabout': [False, False],
            'Station': [False, False],
            'Stop': [True, False],
            'Traffic_Calming': [False, False],
            'Traffic_Signal': [True, False],
            'Sunrise_Sunset': ['Day', 'Night']
        })

    def test_rename_columns_success(self):
        """
        Valida que a função rename_columns renomeia todas as colunas de inglês para português.
        """
        # Executa a renomeação
        renamed_df = rename_columns(self.dummy_data)
        
        # 1. Verifica se o DataFrame retornado não é nulo
        self.assertIsNotNone(renamed_df)
        
        # 2. Verifica se a quantidade de colunas se manteve a mesma
        self.assertEqual(len(renamed_df.columns), len(self.dummy_data.columns))
        
        # 3. Verifica se as colunas esperadas em português estão presentes
        for col_en, col_pt in COLUMNS_MAPPING.items():
            self.assertIn(col_pt, renamed_df.columns)
            self.assertNotIn(col_en, renamed_df.columns)
            
        # 4. Verifica se a integridade dos dados foi mantida
        self.assertEqual(list(renamed_df['Severidade']), [2, 3])
        self.assertEqual(list(renamed_df['Temperatura_F']), [70.5, 65.0])
        self.assertEqual(list(renamed_df['Semaforo']), [True, False])

if __name__ == '__main__':
    unittest.main()
