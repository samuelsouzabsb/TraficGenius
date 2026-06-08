from django.db import models

class Acidente(models.Model):
    """
    Modelo do Django representando um registro de acidente rodoviario.
    Contem todas as 26 variaveis preditoras e a Severidade (classe de destino 1 a 4).
    """
    # Severidade (Target de Previsao: 1 a 4)
    Severidade = models.IntegerField(help_text="Nível de severidade do acidente (1 a 4)")
    
    # Coordenadas Geograficas
    Latitude_Inicial = models.FloatField(help_text="Latitude inicial do local do acidente")
    Longitude_Inicial = models.FloatField(help_text="Longitude inicial do local do acidente")
    
    # Detalhes da via
    Distancia_Milhas = models.FloatField(help_text="Distância da rodovia afetada pelo acidente (em milhas)")
    
    # Variaveis Meteorologicas
    Temperatura_F = models.FloatField(help_text="Temperatura em Fahrenheit")
    Umidade_Percentual = models.FloatField(help_text="Umidade relativa do ar (%)")
    Pressao_Polegadas = models.FloatField(help_text="Pressão atmosférica em polegadas de mercúrio")
    Visibilidade_Milhas = models.FloatField(help_text="Visibilidade em milhas")
    Velocidade_Vento_Mph = models.FloatField(help_text="Velocidade do vento em MPH")
    Precipitacao_Polegadas = models.FloatField(help_text="Acúmulo de precipitação em polegadas")
    
    # Variaveis de Infraestrutura (Flags booleanas)
    Comodidade = models.BooleanField(default=False, help_text="Presença de comodidades/áreas de descanso")
    Lombada = models.BooleanField(default=False, help_text="Presença de lombada física")
    Cruzamento = models.BooleanField(default=False, help_text="Presença de cruzamento")
    Preferencia = models.BooleanField(default=False, help_text="Sinalização de Dê a Preferência")
    Juncao = models.BooleanField(default=False, help_text="Presença de entroncamento/junção")
    Sem_Saida = models.BooleanField(default=False, help_text="Via sem saída")
    Via_Ferrea = models.BooleanField(default=False, help_text="Proximidade de trilhos de trem")
    Rotatoria = models.BooleanField(default=False, help_text="Presença de rotatória")
    Estacao = models.BooleanField(default=False, help_text="Proximidade de estação de transporte público")
    Pare = models.BooleanField(default=False, help_text="Sinalização de Pare (Stop)")
    Redutor_Velocidade = models.BooleanField(default=False, help_text="Presença de redutores de velocidade")
    Semaforo = models.BooleanField(default=False, help_text="Presença de semáforo")
    
    # Variaveis Temporais e Espaciais Calculadas
    Hora_do_Dia = models.IntegerField(help_text="Hora do dia em que ocorreu (0 a 23)")
    Dia_da_Semana = models.IntegerField(help_text="Dia da semana (0 = Segunda, 6 = Domingo)")
    Mes = models.IntegerField(help_text="Mês do ano (1 a 12)")
    Horario_Pico = models.BooleanField(default=False, help_text="Flag indicando se ocorreu em horário de pico de trânsito")
    Cluster_Espacial = models.IntegerField(help_text="ID do cluster geográfico (0 a 19)")

    class Meta:
        verbose_name = "Acidente"
        verbose_name_plural = "Acidentes"
        # Otimiza buscas por regiao geodésica e por severidade
        indexes = [
            models.Index(fields=['Latitude_Inicial', 'Longitude_Inicial'], name='idx_coords'),
            models.Index(fields=['Severidade'], name='idx_severity'),
        ]

    def __str__(self):
        return f"Acidente {self.id} - Severidade {self.Severidade} em ({self.Latitude_Inicial}, {self.Longitude_Inicial})"
