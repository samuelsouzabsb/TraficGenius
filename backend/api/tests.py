from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from unittest.mock import patch
import numpy as np

from .models import Acidente

class AcidenteAPITests(APITestCase):

    def setUp(self):
        # Cria alguns acidentes de teste no banco de dados em memoria
        self.acidente_grave = Acidente.objects.create(
            Severidade=4,
            Latitude_Inicial=40.7128,
            Longitude_Inicial=-74.0060,
            Distancia_Milhas=1.2,
            Temperatura_F=65.0,
            Umidade_Percentual=70.0,
            Pressao_Polegadas=29.92,
            Visibilidade_Milhas=10.0,
            Velocidade_Vento_Mph=8.0,
            Precipitacao_Polegadas=0.0,
            Comodidade=False,
            Lombada=False,
            Cruzamento=True,
            Preferencia=False,
            Juncao=False,
            Sem_Saida=False,
            Via_Ferrea=False,
            Rotatoria=False,
            Estacao=True,
            Pare=False,
            Redutor_Velocidade=False,
            Semaforo=True,
            Hora_do_Dia=14,
            Dia_da_Semana=2,
            Mes=5,
            Horario_Pico=False,
            Cluster_Espacial=3
        )
        
        self.acidente_leve = Acidente.objects.create(
            Severidade=2,
            Latitude_Inicial=34.0522,
            Longitude_Inicial=-118.2437,
            Distancia_Milhas=0.1,
            Temperatura_F=75.0,
            Umidade_Percentual=40.0,
            Pressao_Polegadas=29.95,
            Visibilidade_Milhas=1.5, # Visibilidade baixa para testar SHAP
            Velocidade_Vento_Mph=5.0,
            Precipitacao_Polegadas=0.1, # Com chuva
            Comodidade=False,
            Lombada=False,
            Cruzamento=False,
            Preferencia=False,
            Juncao=True,
            Sem_Saida=False,
            Via_Ferrea=False,
            Rotatoria=False,
            Estacao=False,
            Pare=False,
            Redutor_Velocidade=False,
            Semaforo=False,
            Hora_do_Dia=8, # Hora de pico
            Dia_da_Semana=0,
            Mes=11,
            Horario_Pico=True,
            Cluster_Espacial=0
        )

    def test_list_acidentes(self):
        """Testa se a listagem paginada de acidentes funciona corretamente."""
        url = reverse('acidente-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('results', response.data)
        self.assertEqual(len(response.data['results']), 2)

    def test_list_acidentes_bbox_filtering(self):
        """Testa o filtro de Bounding Box utilizado pelo Leaflet."""
        url = reverse('acidente-list')
        # Filtra apenas a regiao de NY (Nova York)
        response = self.client.get(url, {'in_bbox': '-75,40,-73,42'})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['results']), 1)
        self.assertEqual(response.data['results'][0]['id'], self.acidente_grave.id)

    def test_retrieve_acidente_detail_with_shap(self):
        """Testa se o detalhe do acidente retorna os dados e os valores SHAP simulados."""
        url = reverse('acidente-detail', args=[self.acidente_leve.id])
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('shap_values', response.data)
        # O acidente leve tem chuva, visibilidade baixa e hora de pico, entao deve ter esses SHAP
        shap_features = [item['feature'] for item in response.data['shap_values']]
        self.assertIn('Visibilidade Milhas', shap_features)
        self.assertIn('Precipitacao Polegadas', shap_features)
        self.assertIn('Horario Pico', shap_features)

    def test_dashboard_stats(self):
        """Testa se a view de estatisticas do dashboard retorna a estrutura JSON esperada."""
        url = reverse('dashboard-stats')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        # Valida chaves de primeiro nivel
        self.assertIn('kpis', response.data)
        self.assertIn('shap_data', response.data)
        self.assertIn('time_data', response.data)
        self.assertIn('distribution', response.data)
        self.assertIn('map_clusters', response.data)
        
        # Valida KPIs calculados
        self.assertEqual(response.data['kpis']['max_severity'], 4)
        self.assertEqual(response.data['kpis']['avg_visibility'], 5.75) # (10 + 1.5)/2
        
        # Valida distribuicao de severidade
        self.assertEqual(response.data['distribution']['minor'], 0)
        self.assertEqual(response.data['distribution']['moderate'], 1)
        self.assertEqual(response.data['distribution']['severe'], 0)
        self.assertEqual(response.data['distribution']['fatal'], 1)

    @patch('api.views.get_ml_model_and_scaler')
    def test_predict_severity_endpoint(self, mock_get_model):
        """Testa o endpoint de previsao usando um mock do modelo XGBoost."""
        # Cria mocks do modelo e scaler para evitar carregar do disco nos testes
        class MockModel:
            def predict(self, X):
                return np.array([3]) # Preve classe 3 (Severidade Grau 4)
            def predict_proba(self, X):
                return np.array([[0.05, 0.1, 0.15, 0.7]])

        class MockScaler:
            def transform(self, X):
                return X

        mock_get_model.return_value = (MockModel(), MockScaler())
        
        url = reverse('predict')
        payload = {
            "Latitude_Inicial": 40.7128,
            "Longitude_Inicial": -74.0060,
            "Distancia_Milhas": 0.5,
            "Temperatura_F": 68.0,
            "Umidade_Percentual": 60.0,
            "Pressao_Polegadas": 29.92,
            "Visibilidade_Milhas": 10.0,
            "Velocidade_Vento_Mph": 7.0,
            "Precipitacao_Polegadas": 0.0,
            "Comodidade": False,
            "Lombada": False,
            "Cruzamento": True,
            "Preferencia": False,
            "Juncao": False,
            "Sem_Saida": False,
            "Via_Ferrea": False,
            "Rotatoria": False,
            "Estacao": False,
            "Pare": False,
            "Redutor_Velocidade": False,
            "Semaforo": True,
            "Hora_do_Dia": 12,
            "Dia_da_Semana": 3,
            "Mes": 6,
            "Horario_Pico": False,
            "Cluster_Espacial": 5
        }
        
        response = self.client.post(url, payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['status'], 'success')
        self.assertEqual(response.data['predicted_severity'], 4)
        self.assertEqual(response.data['probabilities']['Grau 4'], 70.0)
