from rest_framework import serializers
from .models import Acidente

class AcidenteSerializer(serializers.ModelSerializer):
    """
    Serializador para o modelo de Acidente. Retorna os dados completos do banco de dados.
    """
    class Meta:
        model = Acidente
        fields = '__all__'


class PredictionInputSerializer(serializers.Serializer):
    """
    Serializador responsável por validar as entradas para a predição de severidade em tempo real.
    Contém as 26 variáveis exigidas pelo modelo XGBoost e scaler.
    """
    Latitude_Inicial = serializers.FloatField(required=True)
    Longitude_Inicial = serializers.FloatField(required=True)
    Distancia_Milhas = serializers.FloatField(required=True)
    Temperatura_F = serializers.FloatField(required=True)
    Umidade_Percentual = serializers.FloatField(required=True)
    Pressao_Polegadas = serializers.FloatField(required=True)
    Visibilidade_Milhas = serializers.FloatField(required=True)
    Velocidade_Vento_Mph = serializers.FloatField(required=True)
    Precipitacao_Polegadas = serializers.FloatField(required=True)
    
    # Flags de infraestrutura
    Comodidade = serializers.BooleanField(required=True)
    Lombada = serializers.BooleanField(required=True)
    Cruzamento = serializers.BooleanField(required=True)
    Preferencia = serializers.BooleanField(required=True)
    Juncao = serializers.BooleanField(required=True)
    Sem_Saida = serializers.BooleanField(required=True)
    Via_Ferrea = serializers.BooleanField(required=True)
    Rotatoria = serializers.BooleanField(required=True)
    Estacao = serializers.BooleanField(required=True)
    Pare = serializers.BooleanField(required=True)
    Redutor_Velocidade = serializers.BooleanField(required=True)
    Semaforo = serializers.BooleanField(required=True)
    
    # Flags temporais e espaciais
    Hora_do_Dia = serializers.IntegerField(required=True, min_value=0, max_value=23)
    Dia_da_Semana = serializers.IntegerField(required=True, min_value=0, max_value=6)
    Mes = serializers.IntegerField(required=True, min_value=1, max_value=12)
    Horario_Pico = serializers.BooleanField(required=True)
    Cluster_Espacial = serializers.IntegerField(required=True, min_value=0, max_value=19)
