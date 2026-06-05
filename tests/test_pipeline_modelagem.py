# -*- coding: utf-8 -*-
"""
Testes Unitários do Pipeline Fase 3 a 5 (Modelagem Preditiva)
Este script valida os componentes auxiliares de modelagem, como cálculo de baselines ao acaso
e criação do gráfico comparativo neon de performance.

Dicas de Inglês (English Tips):
- 'Mock data' = Dados simulados/fictícios para testes.
- 'Chance criteria' = Critérios de classificação ao acaso.
- 'Neon color palette' = Paleta de cores neon.
"""

import unittest
import pandas as pd
import numpy as np
import os
import shutil
import tempfile
from pipeline_fase3a5_modelagem import get_chance_criteria, plot_model_comparison

class TestPipelineModelagem(unittest.TestCase):
    
    def setUp(self):
        """
        Configura o ambiente de testes criando um diretório temporário para saídas gráficas.
        """
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """
        Remove o diretório temporário após a execução dos testes.
        """
        shutil.rmtree(self.test_dir)
        
    def test_get_chance_criteria(self):
        """
        Verifica se a lógica de cálculo de chance (C.Max e C.Prop) está estatisticamente correta.
        """
        # Criando distribuição de classes desbalanceadas controladas
        # 60% classe 1 (majoritária), 30% classe 2, 10% classe 3
        y_mock = pd.Series([1]*60 + [2]*30 + [3]*10)
        
        c_max, c_prop = get_chance_criteria(y_mock)
        
        # C.Max deve ser 60% (classe majoritária)
        self.assertAlmostEqual(c_max, 60.0)
        
        # C.Prop deve ser 0.6^2 + 0.3^2 + 0.1^2 = 0.36 + 0.09 + 0.01 = 0.46 (46%)
        self.assertAlmostEqual(c_prop, 46.0)

    def test_plot_model_comparison(self):
        """
        Valida se a função plot_model_comparison cria a imagem do gráfico comparativo neon.
        """
        mock_metrics = {
            "XGBoost (Tuned)": {"accuracy": 85.5, "f1_macro": 84.0},
            "CNN 1D": {"accuracy": 80.0, "f1_macro": 78.5},
            "Random Forest": {"accuracy": 77.0, "f1_macro": 75.0},
            "Regressão Logística": {"accuracy": 72.0, "f1_macro": 69.5}
        }
        
        # Executa plot
        plot_model_comparison(mock_metrics, self.test_dir)
        
        # Caminho esperado do arquivo de saída
        expected_path = os.path.join(self.test_dir, "comparativo_performance_modelos.png")
        
        # O arquivo deve ter sido criado
        self.assertTrue(os.path.exists(expected_path))
        # O tamanho do arquivo gráfico deve ser superior a zero
        self.assertTrue(os.path.getsize(expected_path) > 0)

if __name__ == '__main__':
    unittest.main()
