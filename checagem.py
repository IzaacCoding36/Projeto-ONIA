"""
Script de verificação de resultados ONIA
Verifica se o arquivo de resultados tem a estrutura correta
"""

import pandas as pd
import os
import sys
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verificar_resultado(arquivo='resultado.csv'):
    """
    Verifica se o arquivo de resultados está correto.
    
    Args:
        arquivo (str): Caminho para o arquivo de resultados
    
    Returns:
        bool: True se válido, False caso contrário
    """
    try:
        if not os.path.exists(arquivo):
            logger.error(f"Arquivo {arquivo} não encontrado")
            return False
        
        # Carregar o arquivo
        logger.info(f"Verificando arquivo: {arquivo}")
        df = pd.read_csv(arquivo)
        
        # Verificar número de linhas
        num_linhas = len(df)
        logger.info(f"Número de linhas: {num_linhas}")
        
        # Verificar colunas
        colunas_esperadas = ['id', 'target']
        if not all(col in df.columns for col in colunas_esperadas):
            logger.error(f"Colunas esperadas: {colunas_esperadas}, encontradas: {list(df.columns)}")
            return False
        
        logger.info(f"Colunas corretas: {list(df.columns)}")
        
        # Verificar se há valores nulos
        valores_nulos = df.isnull().sum().sum()
        if valores_nulos > 0:
            logger.warning(f"Encontrados {valores_nulos} valores nulos")
        else:
            logger.info("Nenhum valor nulo encontrado")
        
        # Verificar tipos de dados
        logger.info(f"Tipo da coluna 'id': {df['id'].dtype}")
        logger.info(f"Tipo da coluna 'target': {df['target'].dtype}")
        
        # Mostrar estatísticas das classes
        distribuicao = df['target'].value_counts().sort_index()
        logger.info("Distribuição das classes:")
        for classe, count in distribuicao.items():
            logger.info(f"  Classe {classe}: {count} ({count/len(df)*100:.1f}%)")
        
        # Verificar se todas as classes são números inteiros
        if not df['target'].dtype in ['int64', 'int32']:
            logger.warning(f"Target deveria ser inteiro, mas é: {df['target'].dtype}")
        
        # Mostrar primeiras linhas
        logger.info("Primeiras 5 linhas:")
        for i, row in df.head().iterrows():
            logger.info(f"  {row['id']},{row['target']}")
        
        logger.info("✅ Verificação concluída com sucesso!")
        return True
        
    except Exception as e:
        logger.error(f"Erro durante verificação: {e}")
        return False

if __name__ == "__main__":
    sucesso = verificar_resultado()
    if not sucesso:
        sys.exit(1)