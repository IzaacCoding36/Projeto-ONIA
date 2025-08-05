[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Orbitron&weight=500&size=17&pause=1000&color=FFFFFF&background=0B004B&center=true&vCenter=true&width=1000&height=100&lines=Primeira+Olimp%C3%ADada+Nacional+de+Intelig%C3%AAncia+Artificial;ONIA;Projeto+Finalista)](https://git.io/typing-svg)

---

# Projeto ONIA

Esse repositório será utilizado para a publicação e desenvolvimento do meu projeto para a primeira Olimpíada Nacional de Inteligência Artificial (ONIA) de 2025.

## 📋 Estrutura do Projeto

- `modelo-xgb-classifier.py` - Script principal original
- `modelo_xgb_classifier_v2.py` - Versão melhorada e modular
- `train.py` - Script de treinamento com configurações flexíveis
- `checagem.py` - Script de verificação de resultados
- `config.py` - Arquivo de configurações
- `requirements.txt` - Dependências do projeto
- `templates/` - Dados de treino e teste

## 🚀 Como Usar

### Instalação das Dependências
```bash
pip install -r requirements.txt
```

### Execução Simples
```bash
# Usar configurações padrão
python modelo_xgb_classifier_v2.py

# Ou usar o script original melhorado
python modelo-xgb-classifier.py
```

### Execução com Configurações Personalizadas
```bash
# Exemplo com parâmetros diferentes
python train.py --n-estimators 1000 --max-depth 15 --learning-rate 0.05

# Sem normalização dos dados
python train.py --no-scaling

# Usando validação de 20%
python train.py --validation-size 0.2
```

### Verificação dos Resultados
```bash
python checagem.py
```

## 🔧 Melhorias Implementadas

### ✅ Portabilidade
- Removidos caminhos hardcodados
- Suporte a diferentes diretórios de dados
- Compatibilidade com diferentes sistemas operacionais

### ✅ Tratamento de Erros
- Validação de entrada de dados
- Tratamento de exceções robusto
- Mensagens de erro informativas

### ✅ Logging e Monitoramento
- Sistema de logging detalhado
- Relatórios de progresso em tempo real
- Métricas de desempenho detalhadas

### ✅ Configuração Flexível
- Arquivo de configuração centralizado
- Parâmetros customizáveis via linha de comando
- Diferentes opções de normalização

### ✅ Código Modular
- Funções bem organizadas e reutilizáveis
- Separação de responsabilidades
- Documentação inline completa

### ✅ Verificação Aprimorada
- Script de verificação robusto
- Validação de estrutura de dados
- Estatísticas detalhadas dos resultados

## 📊 No que se baseia esse projeto?

- Esse projeto baseia-se no treinamento de um modelo de classificação de dados para aprendizagem de máquina com algorítmos. Onde temos dois arquivos com dados .csv para treinar o modelo de algorítmo e trazer previsões de resultados com precisão calculada em Medida-F.
- Isso pode ser feito com vários modelos de classificação, como DecisionTree, RandomForest, K-Nearest Neighbours, Naïve Bayes, Support Vector Machine e o XGBoost, geralmente usado em competições.
- A Medida-F é a média harmônica entre as métricas de Precisão e Revocação. Em outras palavras, a Medida-F é uma métrica que avalia o desempenho de um modelo preditivo de modo a trazer um número único que indique a sua qualidade geral.
- Todos esses conceitos são aplicados em um código python, com os softwares Scikit-Learn ou Orange, com bibliotecas diversificadas para aprendizado de máquina.
- O objetivo de tudo isso é poder treinar um modelo que apresente resultados com um excelente desempenho, medido na medida-f entre 0 e 1, quanto maior for o resultado da medida-f, mais próximo da resposta verídica o seu resultado estará.

## 🎯 Resultados Obtidos

**Medida-F**: ~0.78-0.79 (79% de precisão geral)

**Distribuição das Classes Previstas**:
- Classe 0: ~55% das amostras
- Classe 1: ~14% das amostras  
- Classe 2: ~12% das amostras
- Classe 3: ~5% das amostras
- Classe 4: ~13% das amostras

## 📈 Algoritmo Escolhido: XGBoost

Escolhi o XGBoost por sua precisão em corrigir erros iterativamente. Usei normalização e ajustes para maximizar o desempenho nas 13 features, necessário para ter um resultado com maior precisão para lidar com dados complexos como os fornecidos para essa fase.

**Parâmetros Utilizados**:
- `n_estimators`: 500 árvores
- `max_depth`: 20 (profundidade máxima)
- `learning_rate`: 0.1 (taxa de aprendizado)
- Normalização com `StandardScaler`
- Validação estratificada 90%/10%
