# -*- coding: utf-8 -*-
"""
Módulo de Consolidação de Base de Dados (Dataset Consolidation Module)
Este script é responsável por ler arquivos de dados fragmentados no formato Parquet
e unificá-los em um único arquivo consolidado de forma otimizada para o consumo do pipeline.

Dicas de Inglês (English Tips):
- 'Consolidation' significa unificação ou consolidação.
- 'Row groups' se refere a blocos de linhas de dados armazenados dentro de um arquivo Parquet.
- 'Heap' é a região da memória usada para alocação dinâmica.
- 'Memory leaks' significa vazamento de memória.
- 'Chunk' significa um pedaço ou bloco de dados.
"""

import os
import glob
import time
import pyarrow.parquet as pq
import pyarrow as pa

def consolidate_parquet_files(input_dir, output_file):
    """
    Consolida múltiplos arquivos parquet fragmentados (train-*.parquet) em um único arquivo de forma eficiente.
    Lê os dados em blocos (row groups) usando PyArrow para evitar estouro de memória RAM (Out of Memory - OOM).
    
    Parâmetros (Parameters):
    - input_dir (str): Diretório contendo os arquivos fragmentados de entrada.
    - output_file (str): Caminho absoluto do arquivo único consolidado que será gerado.
    
    Retorno (Returns):
    - total_rows (int): O número total de linhas unificadas.
    """
    # Procura arquivos train-*.parquet no diretório usando expressão de busca (globbing pattern)
    # Exemplo de arquivo buscado: c:\...\dataset\train-00000.parquet
    search_pattern = os.path.join(input_dir, "train-*.parquet")
    files = sorted(glob.glob(search_pattern))
    
    if not files:
        # Se não houver train-*.parquet, tenta achar qualquer parquet no diretório para viabilizar testes ou flexibilidade.
        # Filtra o próprio arquivo de saída se ele já existir para não entrar em loop infinito de leitura/escrita.
        search_pattern_alt = os.path.join(input_dir, "*.parquet")
        files = sorted([f for f in glob.glob(search_pattern_alt) if os.path.basename(f) != os.path.basename(output_file)])
        
    if not files:
        # Se nenhum arquivo for localizado após ambas as tentativas, lança um erro de valor
        raise ValueError(f"Nenhum arquivo Parquet encontrado no diretório: {input_dir}")
        
    print(f"Encontrados {len(files)} arquivos para consolidação.")
    
    writer = None  # Variável para referenciar o gravador de Parquet (ParquetWriter)
    total_rows = 0  # Contador acumulado do número de linhas gravadas
    start_time = time.time()  # Marca o tempo inicial para medição de performance (performance benchmark)
    
    try:
        # Loop para iterar em cada arquivo fragmentado encontrado
        for idx, f in enumerate(files):
            print(f"[{idx+1}/{len(files)}] Processando {os.path.basename(f)}...")
            
            # Abre o arquivo Parquet sem carregar todo o seu conteúdo na RAM imediatamente
            # Isso é chamado de carregamento preguiçoso (lazy loading)
            pf = pq.ParquetFile(f)
            
            # Lê cada grupo de linhas (row group) individualmente
            # Um 'row group' é uma partição lógica horizontal dentro do arquivo Parquet
            for rg_idx in range(pf.num_row_groups):
                # Lê as linhas do bloco específico na forma de uma Tabela PyArrow (PyArrow Table)
                table = pf.read_row_group(rg_idx)
                
                # Inicializa o gravador (ParquetWriter) com o schema (estrutura de dados) do primeiro bloco lido
                if writer is None:
                    # Garante que a pasta pai do arquivo de destino exista no disco
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    # Cria o escritor de Parquet configurando compressão 'snappy' (rápida e eficiente)
                    writer = pq.ParquetWriter(output_file, table.schema, compression='snappy')
                
                # Grava a tabela lida no arquivo consolidado de saída
                writer.write_table(table)
                total_rows += table.num_rows
                
    finally:
        # Garante o fechamento correto do arquivo físico de saída mesmo se ocorrer alguma exceção no meio do loop
        # Isso evita corrupção do arquivo e vazamento de recursos (resource leaks)
        if writer is not None:
            writer.close()
            
    # Calcula o tempo total transcorrido
    elapsed_time = time.time() - start_time
    print(f"Consolidação concluída com sucesso em {elapsed_time:.2f} segundos!")
    print(f"Arquivo gerado: {output_file}")
    print(f"Total de linhas consolidadas: {total_rows}")
    return total_rows

if __name__ == "__main__":
    # Caminho do projeto dinamicamente resolvido
    project_root = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(project_root, "dataset")
    output_parquet = os.path.join(dataset_dir, "dataset_consolidado.parquet")
    
    print("Iniciando processo de consolidação de bases...")
    try:
        # Executa a função principal de consolidação
        total_rows = consolidate_parquet_files(dataset_dir, output_parquet)
        
        # Validação extra pós-processamento para garantir a integridade dos dados (data integrity check)
        if os.path.exists(output_parquet):
            meta = pq.read_metadata(output_parquet)
            print("\n--- Validação do Arquivo Consolidado ---")
            print(f"Arquivo existe: Sim")
            print(f"Número de Linhas no Metadado: {meta.num_rows}")
            print(f"Número de Colunas: {meta.num_columns}")
            print(f"Número de Grupos de Linhas (Row Groups): {meta.num_row_groups}")
            
            # Compara se o número de linhas no disco bate com a soma das linhas lidas
            if meta.num_rows == total_rows:
                print("Validação bem-sucedida! O número de linhas gravadas coincide com o metadado.")
            else:
                print("Aviso: Divergência detectada no número de linhas gravadas vs metadado.")
    except Exception as e:
        print(f"Erro durante a consolidação: {e}")
