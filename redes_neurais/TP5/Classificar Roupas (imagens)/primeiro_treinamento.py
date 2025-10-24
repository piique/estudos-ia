import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import warnings
import io
import os
from contextlib import redirect_stdout
import matplotlib.pyplot as plt

# Silenciar avisos desnecessários
warnings.filterwarnings('ignore')

# --- 1. CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS ---

print("Carregando bases de treino e teste...")
try:
    dados_treino = pd.read_csv('./content/Treino Reconhecimento de Roupas.csv', header=None)
    dados_teste = pd.read_csv('./content/Teste Reconhecimento de Roupas.csv', header=None)
except FileNotFoundError:
    print("\nERRO: Verifique se os arquivos 'Treino Reconhecimento de Roupas.csv' e 'Teste Reconhecimento de Roupas.csv' estão na pasta './content/'.")
    exit()

X_treino_df = dados_treino.iloc[:, 1:]
y_treino_series = dados_treino.iloc[:, 0]
X_teste_df = dados_teste.iloc[:, 1:]
y_teste_series = dados_teste.iloc[:, 0]

X_treino = X_treino_df.values
y_treino = y_treino_series.values
X_teste = X_teste_df.values
y_teste = y_teste_series.values

nomes_classes = [
    'Camisetas/Top', 'Calça', 'Suéter', 'Vestidos', 'Casaco',
    'Sandálias', 'Camisas', 'Tênis', 'Bolsa', 'Botas'
]
num_classes = len(nomes_classes)

X_treino = X_treino / 255.0
X_teste = X_teste / 255.0
print("Carregamento e pré-processamento concluídos.")


# --- 2. CONSTRUÇÃO DO MODELO DE REDE NEURAL (COM MELHORIAS) ---

modelo = keras.Sequential([
    keras.layers.Flatten(input_shape=(784,)),
    
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.4), # Dropout de 40% para evitar overfitting
    
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3), # Dropout de 30%
    
    keras.layers.Dense(num_classes, activation='softmax')
])

modelo.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])


# --- 3. TREINAMENTO DO MODELO (COM MELHORIAS) ---

# 3. USANDO CALLBACKS PARA TREINAMENTO INTELIGENTE
callbacks = [
    # Para o treino se a perda na validação (val_loss) não melhorar por 5 épocas seguidas
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1),
    # Salva apenas o melhor modelo encontrado com base na acurácia da validação
    keras.callbacks.ModelCheckpoint('melhor_modelo.keras', save_best_only=True, monitor='val_accuracy', mode='max')
]

print("\nIniciando o treinamento do modelo...")
# Aumentamos as épocas para 50, mas o EarlyStopping provavelmente vai parar antes
resultado_treinamento = modelo.fit(X_treino, y_treino, batch_size=32, epochs=50, 
                                   validation_data=(X_teste, y_teste), verbose=2,
                                   callbacks=callbacks)
print("Treinamento finalizado.")


# --- 4. AVALIAÇÃO E PREPARAÇÃO DOS RESULTADOS ---

print("\nCarregando o melhor modelo salvo e preparando os resultados...")
# Carrega o melhor modelo que foi salvo pelo ModelCheckpoint
modelo = keras.models.load_model('melhor_modelo.keras')

# Avalia o desempenho final
perda_teste, acuracia_teste = modelo.evaluate(X_teste, y_teste, verbose=0)
print(f'\nAcurácia final na base de testes (melhor modelo): {acuracia_teste:.2%}')

# O restante do código para salvar os resultados permanece igual...
string_io_summary = io.StringIO()
with redirect_stdout(string_io_summary):
    modelo.summary()
resumo_modelo_texto = string_io_summary.getvalue()

string_io_details = io.StringIO()
string_io_details.write("TIPO DE TREINAMENTO:\n")
string_io_details.write("   - Tipo: Aprendizado Supervisionado\n")
string_io_details.write("   - Algoritmo: Backpropagation\n")
string_io_details.write(f"   - Otimizador: Adam (configuração padrão)\n")
string_io_details.write(f"   - Função de Perda (Loss): Sparse Categorical Crossentropy\n\n")
string_io_details.write("PESOS (Wij) E BIAS (bi) POR CAMADA:\n")
for i, camada in enumerate(modelo.layers):
    if len(camada.get_weights()) > 0:
        pesos, biases = camada.get_weights()
        string_io_details.write(f"\n--- Camada {i}: {camada.name} ---\n")
        string_io_details.write(f"   - Formato da Matriz de Pesos (Wij): {pesos.shape}\n")
        string_io_details.write(f"   - Formato do Vetor de Bias (bi): {biases.shape}\n")
detalhes_rede_texto = string_io_details.getvalue()

probabilidades = modelo.predict(X_teste)
indices_preditos = np.argmax(probabilidades, axis=1)
classes_preditas = [nomes_classes[i] for i in indices_preditos]
Y_Resposta = pd.DataFrame(data=classes_preditas, columns=['Previsão de Classe'])


# --- 5. GERAÇÃO E SALVAMENTO DO GRÁFICO DE TREINAMENTO ---

print("Gerando gráfico do histórico de treinamento...")
plt.figure(figsize=(10, 5))
plt.plot(resultado_treinamento.history['accuracy'], label='Acurácia de Treino')
plt.plot(resultado_treinamento.history['val_accuracy'], label='Acurácia de Validação (Teste)')
plt.plot(resultado_treinamento.history['loss'], label='Perda de Treino')
plt.plot(resultado_treinamento.history['val_loss'], label='Perda de Validação (Teste)')
plt.title('Histórico de Treinamento')
plt.ylabel('Acurácia / Perda')
plt.xlabel('Épocas de treinamento')
plt.legend()
plt.grid(True)

os.makedirs('./result', exist_ok=True)
plt.savefig('./result/Historico_de_treinamento.png')
print("Gráfico salvo em ./result/Historico_de_treinamento.png")


# --- 6. SALVANDO OS RESULTADOS EM ARQUIVOS EXTERNOS ---

print("\nSalvando todos os resultados em arquivos na pasta './result'...")
with open('./result/resumo_modelo.txt', 'w', encoding='utf-8') as f:
    f.write(resumo_modelo_texto)
with open('./result/detalhes_rede.txt', 'w', encoding='utf-8') as f:
    f.write(detalhes_rede_texto)
with open('./result/legenda_classes.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(nomes_classes))
Y_Resposta.to_csv('./result/previsoes_finais.csv', index=False)
with open('./result/acuracia_teste.txt', 'w', encoding='utf-8') as f:
    f.write(f"{acuracia_teste:.2%}")
print("\nTodos os arquivos foram salvos com sucesso na pasta './result'.")
