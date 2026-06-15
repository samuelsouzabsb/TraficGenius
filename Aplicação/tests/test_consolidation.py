# -*- coding: utf-8 -*-
"""
Testes Unitários da Consolidação (Consolidation Unit Tests)
Este script implementa testes automatizados utilizando o framework padrão 'unittest' do Python.
O objetivo é garantir a integridade lógica e a estabilidade da função de unificação de arquivos Parquet,
validando que nenhum dado seja corrompido, perdido ou duplicado durante a consolidação.

Dicas de Inglês (English Tips):
- 'Unit tests' significa testes unitários (verificação isolada da menor unidade funcional de código).
- 'setUp' refere-se ao método de preparação executado imediatamente antes de cada função de teste individual.
- 'tearDown' refere-se ao método de limpeza executado imediatamente após cada função de teste individual.
- 'Mock data / Dummy data' são dados simulados estruturados para testar o comportamento do código.
- 'Assert / Assertions' são as afirmações ou checagens lógicas que validam se os resultados coincidem com o esperado.
- 'Tempfile / Temporary directory' é um arquivo ou pasta temporária criada no sistema operacional para fins de testes.
"""

import unittest
import os
import shutil
import tempfile
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from consolidate_dataset import consolidate_parquet_files

class TestParquetConsolidation(unittest.TestCase):
    """
    Classe de teste responsável por verificar a lógica de consolidação de arquivos Parquet.
    Herda de unittest.TestCase para obter acesso a métodos de asserção lógica.
    """
    
    def setUp(self):
        """
        Método de preparação (Setup Phase):
        Cria um diretório temporário isolado e escreve dois arquivos Parquet fictícios (mock datasets)
        com colunas numéricas, de texto e booleanas para validar se a concatenação é perfeita.
        """
        # Cria um diretório de trabalho temporário gerenciado pelo sistema operacional (temp directory)
        self.test_dir = tempfile.mkdtemp()
        
        # Cria dados dummy representativos (fictícios) para o primeiro arquivo
        self.data1 = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10.5, 20.0, 15.2],
            'label': ['A', 'B', 'C'],
            'flag': [True, False, True]
        })
        
        # Cria dados dummy representativos para o segundo arquivo
        self.data2 = pd.DataFrame({
            'id': [4, 5],
            'value': [30.1, 42.0],
            'label': ['D', 'E'],
            'flag': [False, True]
        })
        
        # Define os caminhos dos arquivos parquet dentro do diretório temporário
        self.file1_path = os.path.join(self.test_dir, 'train-00000.parquet')
        self.file2_path = os.path.join(self.test_dir, 'train-00001.parquet')
        
        # Grava os DataFrames em disco no formato Parquet sem incluir o índice das linhas
        self.data1.to_parquet(self.file1_path, index=False)
        self.data2.to_parquet(self.file2_path, index=False)
        
        # Define o caminho planejado do arquivo consolidado de saída
        self.output_file = os.path.join(self.test_dir, 'dataset_consolidado.parquet')

    def tearDown(self):
        """
        Método de desmontagem/limpeza (Teardown Phase):
        Garante a exclusão completa do diretório temporário e de seus arquivos criados.
        Impede o acúmulo de lixo em disco durante execuções recorrentes de testes locais.
        """
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_consolidation_success(self):
        """
        Caso de Teste Principal:
        Valida que múltiplos arquivos fragmentados válidos sejam unificados com sucesso,
        conferindo o total de registros e checando se o conteúdo é exatamente igual à concatenação inicial.
        """
        # Executa a consolidação de teste
        consolidate_parquet_files(self.test_dir, self.output_file)
        
        # Asserção A: O arquivo final foi gerado fisicamente no disco?
        self.assertTrue(os.path.exists(self.output_file))
        
        # Carrega o arquivo final consolidado para inspecionar o conteúdo
        consolidated_df = pd.read_parquet(self.output_file)
        
        # Asserção B: O número total de linhas coincide com a soma dos arquivos de entrada?
        expected_rows = len(self.data1) + len(self.data2)
        self.assertEqual(consolidated_df.shape[0], expected_rows)
        
        # Asserção C: O número de colunas permaneceu idêntico?
        self.assertEqual(consolidated_df.shape[1], self.data1.shape[1])
        
        # Asserção D: Os dados correspondem exatamente à concatenação ideal (sem perda de ordens/valores)?
        # pd.testing.assert_frame_equal faz uma validação profunda célula a célula, incluindo tipos de dados
        original_concat = pd.concat([self.data1, self.data2], ignore_index=True)
        pd.testing.assert_frame_equal(consolidated_df, original_concat)

    def test_consolidation_empty_dir(self):
        """
        Caso de Teste de Exceção:
        Valida que o sistema se comporta corretamente quando acionado em um diretório sem arquivos Parquet,
        levantando uma exceção de valor (ValueError) ou de arquivo não localizado (FileNotFoundError).
        """
        # Cria um novo diretório temporário vazio dedicado
        empty_dir = tempfile.mkdtemp()
        empty_output = os.path.join(empty_dir, 'empty_consolidated.parquet')
        
        try:
            # Asserção: O código deve levantar ValueError ou FileNotFoundError de forma esperada
            with self.assertRaises((FileNotFoundError, ValueError)):
                consolidate_parquet_files(empty_dir, empty_output)
        finally:
            # Garante a limpeza do diretório vazio
            shutil.rmtree(empty_dir)

if __name__ == '__main__':
    unittest.main()
