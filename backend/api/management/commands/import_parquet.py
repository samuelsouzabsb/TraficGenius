import os
import time
import pandas as pd
from django.core.management.base import BaseCommand
from api.models import Acidente

class Command(BaseCommand):
    help = "Importa dados do dataset Parquet limpo para o banco de dados do Django em lotes eficientes."

    def add_arguments(self, parser):
        parser.add_argument(
            '--limit',
            type=int,
            default=None,
            help="Limita o número de linhas a serem importadas (útil para testes rápidos)"
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=25000,
            help="Tamanho do lote para inserção no banco de dados (bulk create)"
        )

    def handle(self, *args, **options):
        limit = options['limit']
        batch_size = options['batch_size']
        
        # Resolve o caminho do parquet relativo a este script
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        parquet_path = os.path.join(project_root, "dataset", "dataset_amostra_limpa_avancado.parquet")
        
        if not os.path.exists(parquet_path):
            self.stdout.write(self.style.ERROR(f"Arquivo parquet nao localizado em: {parquet_path}"))
            return
            
        self.stdout.write(self.style.WARNING(f"Iniciando leitura do Parquet: {os.path.basename(parquet_path)}..."))
        start_time = time.time()
        
        # Colunas de interesse
        columns = [
            'Severidade', 'Latitude_Inicial', 'Longitude_Inicial', 'Distancia_Milhas',
            'Temperatura_F', 'Umidade_Percentual', 'Pressao_Polegadas', 'Visibilidade_Milhas',
            'Velocidade_Vento_Mph', 'Precipitacao_Polegadas', 'Comodidade', 'Lombada',
            'Cruzamento', 'Preferencia', 'Juncao', 'Sem_Saida', 'Via_Ferrea', 'Rotatoria',
            'Estacao', 'Pare', 'Redutor_Velocidade', 'Semaforo', 'Hora_do_Dia',
            'Dia_da_Semana', 'Mes', 'Horario_Pico', 'Cluster_Espacial'
        ]
        
        try:
            # Carrega apenas as colunas necessárias para economizar RAM
            df = pd.read_parquet(parquet_path, columns=columns)
            total_rows = len(df)
            self.stdout.write(self.style.SUCCESS(f"Leitura do Parquet concluída. Total de linhas disponíveis: {total_rows:,}"))
            
            if limit:
                df = df.iloc[:limit]
                total_rows = len(df)
                self.stdout.write(self.style.WARNING(f"Aplicando limite de importacao para as primeiras {total_rows:,} linhas."))
                
            # Limpa registros antigos antes de importar para evitar duplicação em execuções repetidas
            self.stdout.write("Limpando banco de dados de acidentes existentes...")
            Acidente.objects.all().delete()
            
            self.stdout.write(f"Iniciando gravacao no banco de dados em lotes de {batch_size:,}...")
            
            batch = []
            count = 0
            
            for idx, row in df.iterrows():
                # Converte tipos para o Django
                acidente = Acidente(
                    Severidade=int(row['Severidade']),
                    Latitude_Inicial=float(row['Latitude_Inicial']),
                    Longitude_Inicial=float(row['Longitude_Inicial']),
                    Distancia_Milhas=float(row['Distancia_Milhas']),
                    Temperatura_F=float(row['Temperatura_F']),
                    Umidade_Percentual=float(row['Umidade_Percentual']),
                    Pressao_Polegadas=float(row['Pressao_Polegadas']),
                    Visibilidade_Milhas=float(row['Visibilidade_Milhas']),
                    Velocidade_Vento_Mph=float(row['Velocidade_Vento_Mph']),
                    Precipitacao_Polegadas=float(row['Precipitacao_Polegadas']),
                    
                    # Booleans
                    Comodidade=bool(row['Comodidade']),
                    Lombada=bool(row['Lombada']),
                    Cruzamento=bool(row['Cruzamento']),
                    Preferencia=bool(row['Preferencia']),
                    Juncao=bool(row['Juncao']),
                    Sem_Saida=bool(row['Sem_Saida']),
                    Via_Ferrea=bool(row['Via_Ferrea']),
                    Rotatoria=bool(row['Rotatoria']),
                    Estacao=bool(row['Estacao']),
                    Pare=bool(row['Pare']),
                    Redutor_Velocidade=bool(row['Redutor_Velocidade']),
                    Semaforo=bool(row['Semaforo']),
                    
                    # Temporais e espaciais
                    Hora_do_Dia=int(row['Hora_do_Dia']),
                    Dia_da_Semana=int(row['Dia_da_Semana']),
                    Mes=int(row['Mes']),
                    Horario_Pico=bool(row['Horario_Pico']),
                    Cluster_Espacial=int(row['Cluster_Espacial'])
                )
                batch.append(acidente)
                
                if len(batch) >= batch_size:
                    Acidente.objects.bulk_create(batch)
                    count += len(batch)
                    elapsed = time.time() - start_time
                    percent = (count / total_rows) * 100
                    self.stdout.write(f" -> Gravados {count:,} / {total_rows:,} registros ({percent:.1f}%) | Tempo decorrido: {elapsed:.1f}s")
                    batch = []
                    
            # Grava o lote restante
            if batch:
                Acidente.objects.bulk_create(batch)
                count += len(batch)
                
            elapsed_time = time.time() - start_time
            self.stdout.write(self.style.SUCCESS(
                f"\nProcesso finalizado com sucesso!\n"
                f"Total de registros importados: {count:,}\n"
                f"Tempo total transcorrido: {elapsed_time:.2f} segundos."
            ))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Erro durante a importacao: {str(e)}"))
