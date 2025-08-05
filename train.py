"""
Script utilitário para treinamento de modelos ONIA
Permite execução com diferentes configurações
"""

import argparse
import sys
from pathlib import Path

# Importar o modelo principal
import modelo_xgb_classifier_v2 as modelo

def main():
    parser = argparse.ArgumentParser(description='Treinamento de modelo ONIA XGBoost')
    parser.add_argument('--data-dir', default='templates', 
                       help='Diretório contendo os dados (default: templates)')
    parser.add_argument('--output', default='resultado.csv',
                       help='Arquivo de saída (default: resultado.csv)')
    parser.add_argument('--n-estimators', type=int, default=500,
                       help='Número de árvores XGBoost (default: 500)')
    parser.add_argument('--max-depth', type=int, default=20,
                       help='Profundidade máxima das árvores (default: 20)')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                       help='Taxa de aprendizado (default: 0.1)')
    parser.add_argument('--validation-size', type=float, default=0.1,
                       help='Proporção para validação (default: 0.1)')
    parser.add_argument('--random-state', type=int, default=52,
                       help='Semente aleatória (default: 52)')
    parser.add_argument('--no-scaling', action='store_true',
                       help='Desabilitar normalização dos dados')
    
    args = parser.parse_args()
    
    # Verificar se diretório de dados existe
    if not Path(args.data_dir).exists():
        print(f"Erro: Diretório {args.data_dir} não encontrado")
        sys.exit(1)
    
    # Configurar parâmetros
    config = {
        'data_dir': args.data_dir,
        'output_file': args.output,
        'xgboost_params': {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
            'random_state': args.random_state,
            'n_jobs': -1,
            'eval_metric': 'mlogloss'
        },
        'validation_size': args.validation_size,
        'use_scaling': not args.no_scaling
    }
    
    print("=== Configuração do Treinamento ===")
    print(f"Diretório de dados: {config['data_dir']}")
    print(f"Arquivo de saída: {config['output_file']}")
    print(f"Parâmetros XGBoost: {config['xgboost_params']}")
    print(f"Tamanho validação: {config['validation_size']}")
    print(f"Usar normalização: {config['use_scaling']}")
    print("=" * 35)
    
    # Executar treinamento
    try:
        modelo.treinar_modelo(config)
        print("\n✅ Treinamento concluído com sucesso!")
        
    except Exception as e:
        print(f"\n❌ Erro durante treinamento: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()