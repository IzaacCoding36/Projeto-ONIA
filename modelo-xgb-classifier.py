# Importar bibliotecas
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

# 1. Carregar os dados
treino = pd.read_csv('/home/izaaccoding36/olimpiada/treino.csv')
teste = pd.read_csv('/home/izaaccoding36/olimpiada/teste.csv')

# 2. Separar features (X) e alvo (y) do treino
X_treino = treino.drop(columns=['id', 'target'])
y_treino = treino['target']
X_teste = teste.drop(columns=['id'])  # Já separa X_teste aqui

# 2.5. Normalizar os dados
scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)  # Ajusta e transforma o treino
X_teste_scaled = scaler.transform(X_teste)        # Só transforma o teste (sem fit)

# 3. Dividir o treino pra testar o modelo (90% treino, 10% validação)
X_train, X_val, y_train, y_val = train_test_split(X_treino_scaled, y_treino, test_size=0.1, random_state=52)

# 4. Criar e treinar o modelo
modelo = XGBClassifier(n_estimators=500, max_depth=20, learning_rate=0.1, random_state=52)
modelo.fit(X_train, y_train)

# 5. Avaliar no conjunto de validação
previsoes_val = modelo.predict(X_val)
f1 = f1_score(y_val, previsoes_val, average='weighted')
print(f"Medida-F no validação: {f1}")

# 6. Prever no teste
previsoes_teste = modelo.predict(X_teste_scaled)  # Usa os dados normalizados

# 7. Criar o arquivo de resultados
resultado = pd.DataFrame({'id': teste['id'], 'target': previsoes_teste})
resultado.to_csv('resultado.csv', index=False)
print("Arquivo resultado.csv criado com sucesso!")