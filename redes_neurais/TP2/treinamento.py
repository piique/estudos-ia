# Importando pacotes necessários
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')








# Carrega conjunto de dados
dados = pd.read_csv('/content/Rolamento_Treinamento.txt')  
dados.head()   # Ver dados Carregados








# Divide em Entrada e Resposta
X = dados.iloc[:,0:9]    #  Entrada
scaler = StandardScaler()  # Normaliza dos Dados
X = scaler.fit_transform(X)
X    # Ver as entradas Normalizadas








Y1 = dados.iloc[:,9] # Resposta
# Transforma os dados
Y2 = []
for i in range(len(Y1)):
    linha = []
    for j in range(5):
      if (j+1) == Y1[i]:
        linha += [1]
      else:
        linha += [0]
    Y2.append(linha)

Y = pd.DataFrame(data=Y2,columns=['Tipo1', 'Tipo2', 'Tipo3', 'Tipo4', 'Tipo5'])

Y.head() # Ver a Resposta de Treinamento
        






# Define o Modelo
modelo = Sequential()
modelo.add(Dense(9, input_dim=9, activation='relu'))
modelo.add(Dense(5, activation='sigmoid'))
# Compila o modelo
modelo.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Treina o Modelo
resultado = modelo.fit(X, Y, batch_size = 400, epochs = 2000, verbose=0)









modelo.summary()  # Mostra a rede





# Mostra Resultado
import matplotlib.pyplot as plt
plt.plot(resultado.history['loss'])
plt.plot(resultado.history['accuracy'])
plt.title('Histórico de Treinamento')
plt.ylabel('Função de custo')
plt.xlabel('Épocas de treinamento')
plt.legend(['Erro treino', 'Acurácia Treino'])
plt.show()








# Carrega conjunto de teste
testes = pd.read_csv('/content/Rolamento_Teste.txt') 
testes.head()  # Ver testes

Xtestes = testes.iloc[:,0:9] # Entrada dos Testes
Xtestes  # Ver os testes

Xtestes = scaler.fit_transform(Xtestes)  # Normaliza dos Testes
Xtestes







# Testa a rede
Y_predito = modelo.predict(Xtestes)
print("Valores Preditos:",Y_predito)

Y_predito1 = np.array([])
Y_Resposta1 = np.array([])
for i in Y_predito:
    Y_parte = ([1 if max(i)==y else 0 for y in i])
    print(Y_parte)
    Y_predito1 = np.concatenate((Y_predito1,Y_parte))
    for j in range(len(Y_parte)):
      if Y_parte[j]==1:
        Resp = ([j+1])
    
    Y_Resposta1 = np.concatenate((Y_Resposta1,Resp))

Y_Resposta1 # Ver os Resuladados







Y_Resposta = pd.DataFrame(data=Y_Resposta1, dtype=np.int8, columns=['Tipo 1 a 5'])
Y_Resposta  # Ver a Resposta







# Mostra Pesos
for layerNum, layer in enumerate(modelo.layers):
    weights = layer.get_weights()[0]
    biases = layer.get_weights()[1]

    for toNeuronNum, bias in enumerate(biases):
        print(f'{layerNum}B -> L{layerNum+1}N{toNeuronNum}: {bias}')

    for fromNeuronNum, wgt in enumerate(weights):
        for toNeuronNum, wgt2 in enumerate(wgt):
            print(f'L{layerNum}N{fromNeuronNum} \
                  -> L{layerNum+1}N{toNeuronNum} = {wgt2}')





