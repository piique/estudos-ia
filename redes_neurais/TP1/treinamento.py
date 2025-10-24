import tensorflow as tf
print(tf.__version__)

import keras
import pandas as pd
import numpy as np

# import warnings
# warnings.filterwarnings('ignore')




X_treino = np.array(
    [
        [26,1.50], [19,1.70], [18,1.80], [17,1.30], [24,1.30],
        [17,1.20], [15,1.60], [16,1.40], [14,1.45], [23,1.70], [17,1.35],
        [18,1.90], [17,1.75], [25,1.80], [12,1.25], [22,1.50]
    ]
)
Y_treino = np.array( [
         [1,0,0],[0,1,0],[0,1,0],[0,0,1],[1,0,0],
         [0,0,1],[0,1,0],[0,0,1],[0,0,1],[1,0,0],[0,0,1],
         [0,1,0],[0,1,0],[1,0,0],[0,0,1],[1,0,0] ] )
X_teste = np.array(
    [
        [25,1.67], [22,1.47], [23,1.88]
    ]
)
Y_teste = np.array( [[1,0,0],[1,0,0],[1,0,0]] )


X_train = X_treino
X_test = X_teste





X_train





X_test







from keras.models import Sequential
from keras.layers import Dense

modelo = Sequential() # Inicializa a Rede Neural Artificial
# modelo.add(Dense(units = 3, activation = 'tanh', input_dim = X_train.shape[1]))
modelo.add(Dense(units = 3, activation = 'sigmoid', input_dim = X_train.shape[1]))







modelo.compile(loss='mse', optimizer = 'adam', metrics=['mae'])


resultado = modelo.fit(X_train, Y_treino, batch_size = 1, epochs = 1500, validation_data=(X_test, Y_teste))








Y_predito = modelo.predict(X_test)
Y_correto = np.array([])
print("Valores Preditos:",Y_predito)








for i in Y_predito:
    Y_parte = ([1 if max(i)==y else 0 for y in i])
    print(Y_parte)
    Y_correto = np.concatenate((Y_correto,Y_parte))

Y_correto = Y_correto.reshape(Y_predito.shape[0], Y_predito.shape[1])
print("Valores Preditos:",Y_correto)






total = 0
correto = 0
errado = 0
for i in range(Y_correto.shape[0]):
  total=total+1
  if((Y_teste[i,0] == Y_correto[i,0]) and (Y_teste[i,1] == Y_correto[i,1]) and (Y_teste[i,2] == Y_correto[i,2])):
    correto=correto+1
  else:
    errado=errado+1

print("Total " + str(total))
print("Correto " + str(correto))
print("Errado " + str(errado))










import matplotlib.pyplot as plt

plt.plot(resultado.history['loss'])
plt.plot(resultado.history['val_loss'])
plt.title('Histórico de Treinamento')
plt.ylabel('Função de custo')
plt.xlabel('Épocas de treinamento')
plt.legend(['Erro treino', 'Erro teste'])
plt.show()







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


