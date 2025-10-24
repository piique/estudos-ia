import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

dados = pd.read_csv('./content/Motores_Treinamento.txt', header=None)

# Ver dados
dados.head()

X1 = dados.iloc[:,0:7] #Entrada

X = X1.values

XX = pd.DataFrame(data=X,columns=X1.columns)
XX.head()


# Normnaliza os dados
scaler = StandardScaler()
XX = scaler.fit_transform(XX)

Y1 = dados.iloc[:,7] #Resposta
Y = Y1.values

# Transforma os dados
Y2 = []
for i in range(len(Y)):
    linha = []
    for j in range(3):
      if (j+1) == Y[i]:
        linha += [1]
      else:
        linha += [0]
    Y2.append(linha)


YY = pd.DataFrame(data=Y2,columns=['Tipo1', 'Tipo2', 'Tipo3'])

# Ver Y
YY.head()


# Define o Modelo
modelo = Sequential()
modelo.add(Dense(14, input_dim=7, activation='relu'))
modelo.add(Dense(3, activation='sigmoid'))

# Compila o modelo
modelo.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treina o Modelo
resultado = modelo.fit(XX, YY, batch_size = 32, epochs = 8000, verbose=0)


# Mostra a rede
modelo.summary()


plt.plot(resultado.history['loss'])
plt.plot(resultado.history['accuracy'])
plt.title('Histórico de Treinamento')
plt.ylabel('Função de custo')
plt.xlabel('Épocas de treinamento')
plt.legend(['Erro treino', 'Acurácia Treino'])
plt.show()

# Carrega conjunto de teste
testes = pd.read_csv('./content/Motores_Teste.txt', header=None)

# Ver testes
testes.head()


# Divide em Entrada
Xtestes1 = testes.iloc[:,0:7] #Entrada

# Ver os testes
print("Dados de Teste: ")
print(Xtestes1)


Xt = Xtestes1.values

Xtestes = pd.DataFrame(data=Xt,columns=testes.columns)


# Normaliza os dados de teste
Xtestes = scaler.fit_transform(Xtestes)

# Testa a rede
Y_predito = modelo.predict(Xtestes)
print("Valores Preditos:",Y_predito)

Y_predito1 = np.array([])
Yr = np.array([])
for i in Y_predito:
    Y_parte = ([1 if max(i)==y else 0 for y in i])
    print(Y_parte)
    Y_predito1 = np.concatenate((Y_predito1,Y_parte))
    for j in range(len(Y_parte)):
      if Y_parte[j]==1:
        Resp = ([j+1])

    Yr = np.concatenate((Yr,Resp))

YYr = np.array([])
for i in range(len(Yr)):
  if Yr[i] == 3:
    r = "3"
  if Yr[i] == 2:
    r = "2"
  if Yr[i] == 1:
    r = "1"

  YYr = np.concatenate((YYr,[r]))


# Ver Resposta
Y_Resposta = pd.DataFrame(data=YYr, columns=['Tipo de Motores'])
print("Resposta: ")
print(Y_Resposta)



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
