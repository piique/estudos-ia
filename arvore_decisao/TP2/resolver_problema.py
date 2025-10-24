# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score

# --- CONFIGURAÇÕES ---
RESULT_DIR = "./result"

# --- CRIAÇÃO DO DIRETÓRIO DE RESULTADOS ---
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
print(f"Diretório '{RESULT_DIR}' pronto para receber os resultados.")


# --- FUNÇÃO PARA RESOLVER O EXERCÍCIO 1: ROLAMENTOS ---
def resolver_exercicio_1():
    """
    Resolve o exercício de classificação de rolamentos usando arquivos de treino e teste.
    """
    print("\n--- Iniciando Exercício 1: Classificação de Rolamentos ---")

    # 1. Carregar os dados de TREINAMENTO
    caminho_treino = os.path.join(
        "content", "Classificacao de Mancais de Rolamentos (Treinamento).csv"
    )
    try:
        dados_treino = pd.read_csv(caminho_treino)
    except FileNotFoundError:
        print(f"ERRO: Arquivo de treinamento '{caminho_treino}' não encontrado.")
        return

    # 2. Separar Features (X) e Target (Y) para o treinamento
    previsores_nomes = list(dados_treino.columns[:-1])
    X_treino = dados_treino.iloc[:, :-1]
    Y_treino = dados_treino.iloc[:, -1]

    # 3. Criar e Treinar o Classificador
    arvore = DecisionTreeClassifier(criterion="entropy")
    arvore.fit(X_treino, Y_treino)

    acuracia_treino = arvore.score(X_treino, Y_treino)
    print(f"Acurácia do modelo nos dados de treino: {acuracia_treino*100:.2f}%")

    # 4. Gerar e Salvar os Resultados Visuais e Regras

    # a) Construir a árvore (Salvar como imagem)
    fig, ax = plt.subplots(figsize=(20, 12))
    plot_tree(
        arvore,
        feature_names=previsores_nomes,
        class_names=[str(c) for c in arvore.classes_],
        filled=True,
        rounded=True,
        fontsize=10,
    )
    caminho_imagem = os.path.join(RESULT_DIR, "ex1_arvore.png")
    plt.savefig(caminho_imagem)
    plt.close(fig)
    print(f"Imagem da árvore salva em: {caminho_imagem}")

    # b) Indicar a regra SE-ENTÃO
    regras = export_text(arvore, feature_names=previsores_nomes)
    caminho_regras = os.path.join(RESULT_DIR, "ex1_regras.txt")
    with open(caminho_regras, "w", encoding="utf-8") as f:
        f.write(regras)
    print(f"Regras SE-ENTÃO salvas em: {caminho_regras}")

    # 5. Carregar dados de TESTE e fazer as classificações
    caminho_teste = os.path.join(
        "content", "Classificacao de Mancais de Rolamentos (Teste).csv"
    )
    try:
        dados_teste = pd.read_csv(caminho_teste)
        X_teste = dados_teste.iloc[:, :-1]
        Y_verdadeiro = dados_teste.iloc[:, -1]
    except FileNotFoundError:
        print(
            f"ERRO: Arquivo de teste '{caminho_teste}' não encontrado. As previsões não serão geradas."
        )
        return

    # Fazer previsões para todo o conjunto de teste
    Y_previsoes = arvore.predict(X_teste)

    # Calcular a acurácia no conjunto de teste
    acuracia_teste = accuracy_score(Y_verdadeiro, Y_previsoes)
    print(f"Acurácia do modelo nos dados de teste: {acuracia_teste*100:.2f}%")

    # Salvar os resultados das previsões em um arquivo
    previsoes_texto = []
    previsoes_texto.append(
        f"Acurácia geral no conjunto de teste: {acuracia_teste*100:.2f}%\n"
    )
    previsoes_texto.append("Previsões para cada instância de teste:")

    for i in range(len(dados_teste)):
        instancia = X_teste.iloc[i].values
        previsao = Y_previsoes[i]
        resultado = f"  - Previsão para {instancia}: Classe '{previsao}' (Valor real: '{Y_verdadeiro.iloc[i]}')"
        previsoes_texto.append(resultado)

    caminho_previsoes = os.path.join(RESULT_DIR, "ex1_previsoes.txt")
    with open(caminho_previsoes, "w", encoding="utf-8") as f:
        f.write("\n".join(previsoes_texto))
    print(f"Previsões do conjunto de teste salvas em: {caminho_previsoes}")
    print("--- Exercício 1 Finalizado ---")


# --- FUNÇÃO PARA RESOLVER O EXERCÍCIO 2: IRIS DATASET ---
def resolver_exercicio_2():
    """
    Resolve o exercício de classificação do dataset Iris.
    """
    print("\n--- Iniciando Exercício 2: Classificação de Flores Íris ---")

    caminho_arquivo = os.path.join("content", "iris.csv")
    try:
        dados = pd.read_csv(caminho_arquivo)
    except FileNotFoundError:
        print(f"ERRO: Arquivo '{caminho_arquivo}' não encontrado.")
        return

    previsores_nomes = list(dados.columns[:-1])
    XX = dados.iloc[:, :-1]
    YY = dados.iloc[:, -1]

    arvore = DecisionTreeClassifier(criterion="entropy")
    arvore.fit(XX, YY)

    acuracia = arvore.score(XX, YY)
    print(f"Acurácia do modelo nos dados de treino: {acuracia*100:.2f}%")

    fig, ax = plt.subplots(figsize=(25, 15))
    plot_tree(
        arvore,
        feature_names=previsores_nomes,
        class_names=arvore.classes_,
        filled=True,
        rounded=True,
        fontsize=10,
    )
    caminho_imagem = os.path.join(RESULT_DIR, "ex2_arvore.png")
    plt.savefig(caminho_imagem)
    plt.close(fig)
    print(f"Imagem da árvore salva em: {caminho_imagem}")

    regras = export_text(arvore, feature_names=previsores_nomes)
    caminho_regras = os.path.join(RESULT_DIR, "ex2_regras.txt")
    with open(caminho_regras, "w", encoding="utf-8") as f:
        f.write(regras)
    print(f"Regras SE-ENTÃO salvas em: {caminho_regras}")

    previsoes_texto = []

    dados_c = [[5.1, 3.5, 1.4, 0.2]]
    previsao_c = arvore.predict(dados_c)
    resultado_c = f"Previsão para [5.1, 3.5, 1.4, 0.2]: Espécie '{previsao_c[0]}'"
    previsoes_texto.append(resultado_c)
    print(resultado_c)

    dados_d = [[6.7, 3.0, 5.2, 2.3]]
    previsao_d = arvore.predict(dados_d)
    resultado_d = f"Previsão para [6.7, 3.0, 5.2, 2.3]: Espécie '{previsao_d[0]}'"
    previsoes_texto.append(resultado_d)
    print(resultado_d)

    caminho_previsoes = os.path.join(RESULT_DIR, "ex2_previsoes.txt")
    with open(caminho_previsoes, "w", encoding="utf-8") as f:
        f.write("\n".join(previsoes_texto))
    print(f"Previsões salvas em: {caminho_previsoes}")
    print("--- Exercício 2 Finalizado ---")


# --- EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    resolver_exercicio_1()
    resolver_exercicio_2()
    print(
        "\nProcesso finalizado. Todos os arquivos de resultado foram salvos na pasta 'result/'."
    )
