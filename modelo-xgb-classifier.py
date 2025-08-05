"""
ONIA - Olimpíada Nacional de Inteligência Artificial
Classificador XGBoost para predição de dados de machine learning

Autor: IzaacCoding36
Versão melhorada com portabilidade e tratamento de erros
"""

# Importar bibliotecas
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

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('modelo_onia.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def carregar_dados(diretorio='templates'):
    """
    Carrega os dados de treino e teste do diretório especificado.
    
    Args:
        diretorio (str): Diretório onde estão os arquivos CSV
    
    Returns:
        tuple: (treino_df, teste_df) ou (None, None) em caso de erro
    """
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

# 1. Carregar os dados
treino, teste = carregar_dados()

if treino is None or teste is None:
    logger.error("Falha ao carregar os dados. Encerrando execução.")
    sys.exit(1)

def validar_dados(treino_df, teste_df):
    """
    Valida se os dados têm a estrutura esperada.
    
    Args:
        treino_df (DataFrame): Dados de treino
        teste_df (DataFrame): Dados de teste
    
    Returns:
        bool: True se válido, False caso contrário
    """
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

# Validar dados
if not validar_dados(treino, teste):
    logger.error("Dados inválidos. Encerrando execução.")
    sys.exit(1)

# 2. Separar features (X) e alvo (y) do treino
try:
    X_treino = treino.drop(columns=['id', 'target'])
    y_treino = treino['target']
    X_teste = teste.drop(columns=['id'])  # Já separa X_teste aqui
    
    logger.info(f"Features de treino: {X_treino.shape}")
    logger.info(f"Target de treino: {y_treino.shape}")
    logger.info(f"Features de teste: {X_teste.shape}")
    
except Exception as e:
    logger.error(f"Erro ao separar features e target: {e}")
    sys.exit(1)

# 2.5. Normalizar os dados
try:
    logger.info("Normalizando dados com StandardScaler...")
    scaler = StandardScaler()
    X_treino_scaled = scaler.fit_transform(X_treino)  # Ajusta e transforma o treino
    X_teste_scaled = scaler.transform(X_teste)        # Só transforma o teste (sem fit)
    logger.info("Normalização concluída")
    
except Exception as e:
    logger.error(f"Erro na normalização dos dados: {e}")
    sys.exit(1)

# 3. Dividir o treino pra testar o modelo (90% treino, 10% validação)
try:
    logger.info("Dividindo dados em treino e validação (90%/10%)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_treino_scaled, y_treino, 
        test_size=0.1, 
        random_state=52,
        stratify=y_treino  # Manter proporção das classes
    )
    logger.info(f"Divisão concluída: {X_train.shape[0]} treino, {X_val.shape[0]} validação")
    
except Exception as e:
    logger.error(f"Erro na divisão dos dados: {e}")
    sys.exit(1)

# 4. Criar e treinar o modelo
def criar_modelo(n_estimators=500, max_depth=20, learning_rate=0.1, random_state=52):
    """
    Cria e configura o modelo XGBoost.
    
    Args:
        n_estimators (int): Número de árvores
        max_depth (int): Profundidade máxima das árvores
        learning_rate (float): Taxa de aprendizado
        random_state (int): Semente para reprodutibilidade
    
    Returns:
        XGBClassifier: Modelo configurado
    """
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        n_jobs=-1,  # Usar todos os cores disponíveis
        eval_metric='mlogloss'  # Métrica para multi-classe
    )

try:
    logger.info("Criando e treinando modelo XGBoost...")
    modelo = criar_modelo()
    modelo.fit(X_train, y_train)
    logger.info("Treinamento concluído!")
    
except Exception as e:
    logger.error(f"Erro no treinamento do modelo: {e}")
    sys.exit(1)

# 5. Avaliar no conjunto de validação
try:
    logger.info("Avaliando modelo no conjunto de validação...")
    previsoes_val = modelo.predict(X_val)
    f1 = f1_score(y_val, previsoes_val, average='weighted')
    
    logger.info(f"Medida-F no conjunto de validação: {f1:.4f}")
    
    # Relatório detalhado de classificação
    logger.info("Relatório de classificação:")
    relatorio = classification_report(y_val, previsoes_val)
    logger.info(f"\n{relatorio}")
    
except Exception as e:
    logger.error(f"Erro na avaliação do modelo: {e}")
    sys.exit(1)

# 6. Prever no teste
try:
    logger.info("Gerando previsões para o conjunto de teste...")
    previsoes_teste = modelo.predict(X_teste_scaled)  # Usa os dados normalizados
    logger.info(f"Previsões geradas para {len(previsoes_teste)} amostras")
    
except Exception as e:
    logger.error(f"Erro na geração de previsões: {e}")
    sys.exit(1)

# 7. Criar o arquivo de resultados
def salvar_resultados(ids_teste, previsoes, arquivo_saida='resultado.csv'):
    """
    Salva as previsões em arquivo CSV.
    
    Args:
        ids_teste (array): IDs das amostras de teste
        previsoes (array): Previsões do modelo
        arquivo_saida (str): Nome do arquivo de saída
    
    Returns:
        bool: True se salvou com sucesso, False caso contrário
    """
    try:
        resultado = pd.DataFrame({'id': ids_teste, 'target': previsoes})
        resultado.to_csv(arquivo_saida, index=False)
        
        logger.info(f"Arquivo {arquivo_saida} criado com sucesso!")
        logger.info(f"Total de previsões: {len(resultado)}")
        
        # Mostrar distribuição das classes previstas
        distribuicao = resultado['target'].value_counts().sort_index()
        logger.info(f"Distribuição das classes previstas:")
        for classe, count in distribuicao.items():
            logger.info(f"  Classe {classe}: {count} amostras ({count/len(resultado)*100:.1f}%)")
        
        return True
        
    except Exception as e:
        logger.error(f"Erro ao salvar resultados: {e}")
        return False

# Salvar resultados
if salvar_resultados(teste['id'], previsoes_teste):
    logger.info("Processo concluído com sucesso!")
else:
    logger.error("Falha ao salvar resultados")
    sys.exit(1)