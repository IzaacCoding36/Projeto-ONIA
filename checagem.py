with open('resultado.csv', 'r') as f:
    linhas = f.readlines()
    print(len(linhas))  # 4501
    print(linhas[0])    # Deve ser "id,target\n" ou similar