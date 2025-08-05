"""
ONIA - Olimpíada Nacional de Inteligência Artificial
Versão modular do classificador XGBoost

Autor: IzaacCoding36
Versão melhorada com funções modulares e configuração flexível
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import os
import sys
import logging
from pathlib import Path
import config

def configurar_logging(log_file=None, level=logging.INFO):
    """Configura o sistema de logging."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=config.LOG_FORMAT,
        handlers=handlers
    )
    return logging.getLogger(__name__)

def carregar_dados(diretorio='templates'):
    """
    Carrega os dados de treino e teste do diretório especificado.
    
    Args:
        diretorio (str): Diretório onde estão os arquivos CSV
    
    Returns:
        tuple: (treino_df, teste_df) ou (None, None) em caso de erro
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Usar paths relativos para portabilidade
        caminho_treino = Path(diretorio) / 'treino.csv'
        caminho_teste = Path(diretorio) / 'teste.csv'
        
        logger.info(f"Carregando dados de treino: {caminho_treino}")
        treino = pd.read_csv(caminho_treino)
        
        logger.info(f"Carregando dados de teste: {caminho_teste}")
        teste = pd.read_csv(caminho_teste)
        
        logger.info(f"Dados carregados: {len(treino)} amostras de treino, {len(teste)} amostras de teste")
        
        return treino, teste
        
    except FileNotFoundError as e:
        logger.error(f"Arquivo não encontrado: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        return None, None

def validar_dados(treino_df, teste_df):
    """
    Valida se os dados têm a estrutura esperada.
    
    Args:
        treino_df (DataFrame): Dados de treino
        teste_df (DataFrame): Dados de teste
    
    Returns:
        bool: True se válido, False caso contrário
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Verificar se colunas essenciais existem
        if 'id' not in treino_df.columns or 'target' not in treino_df.columns:
            logger.error("Arquivo de treino deve conter colunas 'id' e 'target'")
            return False
            
        if 'id' not in teste_df.columns:
            logger.error("Arquivo de teste deve conter coluna 'id'")
            return False
        
        # Verificar se número de features é compatível
        features_treino = treino_df.drop(['id', 'target'], axis=1).columns
        features_teste = teste_df.drop(['id'], axis=1).columns
        
        if len(features_treino) != len(features_teste):
            logger.warning(f"Número de features diferentes: treino={len(features_treino)}, teste={len(features_teste)}")
        
        logger.info(f"Validação concluída: {len(features_treino)} features identificadas")
        return True
        
    except Exception as e:
        logger.error(f"Erro na validação dos dados: {e}")
        return False

def preparar_dados(treino, teste, use_scaling=True, validation_size=0.1, random_state=52):
    """
    Prepara os dados para treinamento.
    
    Args:
        treino (DataFrame): Dados de treino
        teste (DataFrame): Dados de teste
        use_scaling (bool): Se deve normalizar os dados
        validation_size (float): Proporção para validação
        random_state (int): Semente aleatória
    
    Returns:
        tuple: (X_train, X_val, y_train, y_val, X_teste_final, ids_teste, scaler)
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Separar features e target
        X_treino = treino.drop(columns=['id', 'target'])
        y_treino = treino['target']
        X_teste = teste.drop(columns=['id'])
        ids_teste = teste['id']
        
        logger.info(f"Features de treino: {X_treino.shape}")
        logger.info(f"Target de treino: {y_treino.shape}")
        logger.info(f"Features de teste: {X_teste.shape}")
        
        # Normalização (opcional)
        scaler = None
        if use_scaling:
            logger.info("Normalizando dados com StandardScaler...")
            scaler = StandardScaler()
            X_treino_scaled = scaler.fit_transform(X_treino)
            X_teste_scaled = scaler.transform(X_teste)
        else:
            logger.info("Pular normalização...")
            X_treino_scaled = X_treino.values
            X_teste_scaled = X_teste.values
        
        # Dividir dados
        logger.info(f"Dividindo dados em treino e validação ({(1-validation_size)*100:.0f}%/{validation_size*100:.0f}%)...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_treino_scaled, y_treino,
            test_size=validation_size,
            random_state=random_state,
            stratify=y_treino
        )
        
        logger.info(f"Divisão concluída: {X_train.shape[0]} treino, {X_val.shape[0]} validação")
        
        return X_train, X_val, y_train, y_val, X_teste_scaled, ids_teste, scaler
        
    except Exception as e:
        logger.error(f"Erro na preparação dos dados: {e}")
        raise

def treinar_modelo_xgb(X_train, y_train, xgboost_params=None):
    """
    Treina o modelo XGBoost.
    
    Args:
        X_train (array): Features de treino
        y_train (array): Target de treino
        xgboost_params (dict): Parâmetros do XGBoost
    
    Returns:
        XGBClassifier: Modelo treinado
    """
    logger = logging.getLogger(__name__)
    
    if xgboost_params is None:
        xgboost_params = config.XGBOOST_PARAMS
    
    try:
        logger.info("Criando e treinando modelo XGBoost...")
        logger.info(f"Parâmetros: {xgboost_params}")
        
        modelo = XGBClassifier(**xgboost_params)
        modelo.fit(X_train, y_train)
        
        logger.info("Treinamento concluído!")
        return modelo
        
    except Exception as e:
        logger.error(f"Erro no treinamento do modelo: {e}")
        raise

def avaliar_modelo(modelo, X_val, y_val):
    """
    Avalia o modelo no conjunto de validação.
    
    Args:
        modelo: Modelo treinado
        X_val (array): Features de validação
        y_val (array): Target de validação
    
    Returns:
        float: F1-score
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Avaliando modelo no conjunto de validação...")
        previsoes_val = modelo.predict(X_val)
        f1 = f1_score(y_val, previsoes_val, average='weighted')
        
        logger.info(f"Medida-F no conjunto de validação: {f1:.4f}")
        
        # Relatório detalhado
        relatorio = classification_report(y_val, previsoes_val)
        logger.info(f"Relatório de classificação:\n{relatorio}")
        
        return f1
        
    except Exception as e:
        logger.error(f"Erro na avaliação do modelo: {e}")
        raise

def gerar_previsoes(modelo, X_teste, ids_teste, arquivo_saida='resultado.csv'):
    """
    Gera previsões e salva em arquivo.
    
    Args:
        modelo: Modelo treinado
        X_teste (array): Features de teste
        ids_teste (array): IDs de teste
        arquivo_saida (str): Arquivo de saída
    
    Returns:
        bool: True se sucesso
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Gerando previsões para o conjunto de teste...")
        previsoes_teste = modelo.predict(X_teste)
        logger.info(f"Previsões geradas para {len(previsoes_teste)} amostras")
        
        # Salvar resultados
        resultado = pd.DataFrame({'id': ids_teste, 'target': previsoes_teste})
        resultado.to_csv(arquivo_saida, index=False)
        
        logger.info(f"Arquivo {arquivo_saida} criado com sucesso!")
        logger.info(f"Total de previsões: {len(resultado)}")
        
        # Mostrar distribuição
        distribuicao = resultado['target'].value_counts().sort_index()
        logger.info("Distribuição das classes previstas:")
        for classe, count in distribuicao.items():
            logger.info(f"  Classe {classe}: {count} amostras ({count/len(resultado)*100:.1f}%)")
        
        return True
        
    except Exception as e:
        logger.error(f"Erro ao gerar previsões: {e}")
        return False

def treinar_modelo(configuracao=None):
    """
    Função principal para treinamento completo do modelo.
    
    Args:
        configuracao (dict): Configurações personalizadas
    """
    # Configurar logging
    logger = configurar_logging(config.LOG_FILE)
    
    # Usar configuração padrão se não fornecida
    if configuracao is None:
        configuracao = {
            'data_dir': config.DATA_DIR,
            'output_file': config.OUTPUT_FILE,
            'xgboost_params': config.XGBOOST_PARAMS,
            'validation_size': config.VALIDATION_SIZE,
            'use_scaling': config.USE_SCALING
        }
    
    try:
        # 1. Carregar dados
        treino, teste = carregar_dados(configuracao['data_dir'])
        if treino is None or teste is None:
            logger.error("Falha ao carregar os dados. Encerrando execução.")
            return False
        
        # 2. Validar dados
        if not validar_dados(treino, teste):
            logger.error("Dados inválidos. Encerrando execução.")
            return False
        
        # 3. Preparar dados
        X_train, X_val, y_train, y_val, X_teste, ids_teste, scaler = preparar_dados(
            treino, teste,
            use_scaling=configuracao['use_scaling'],
            validation_size=configuracao['validation_size'],
            random_state=configuracao['xgboost_params']['random_state']
        )
        
        # 4. Treinar modelo
        modelo = treinar_modelo_xgb(X_train, y_train, configuracao['xgboost_params'])
        
        # 5. Avaliar modelo
        f1_score_val = avaliar_modelo(modelo, X_val, y_val)
        
        # 6. Gerar previsões
        sucesso = gerar_previsoes(modelo, X_teste, ids_teste, configuracao['output_file'])
        
        if sucesso:
            logger.info("Processo concluído com sucesso!")
            return True
        else:
            logger.error("Falha ao salvar resultados")
            return False
            
    except Exception as e:
        logger.error(f"Erro durante execução: {e}")
        return False

if __name__ == "__main__":
    # Executar com configurações padrão
    sucesso = treinar_modelo()
    if not sucesso:
        sys.exit(1)