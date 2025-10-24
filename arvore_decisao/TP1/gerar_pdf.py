# -*- coding: utf-8 -*-

import os
from fpdf import FPDF
from fpdf.enums import XPos
from dotenv import load_dotenv  # Adicionado para ler o arquivo .env

# --- CARREGAR VARIÁVEIS DE AMBIENTE ---
load_dotenv()  # Carrega as variáveis do arquivo .env

# --- CONFIGURAÇÕES ---
# Pega o nome do aluno da variável de ambiente, com um valor padrão caso não encontre
NOME_ALUNO = os.getenv("NOME", "Seu Nome Completo Aqui")
RESULT_DIR = "./result"
NOME_ARQUIVO_SCRIPT = "resolver_problema.py"  # Nome do script que contém a lógica


def sanitize_text(text):
    """Remove caracteres incompatíveis com a codificação latin-1 do FPDF."""
    return text.encode("latin-1", "ignore").decode("latin-1")


class PDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            self.set_font("Helvetica", "B", 18)
            self.cell(
                0,
                10,
                "Trabalho Prático de IA - Árvore de Decisão",
                border=0,
                new_x=XPos.LMARGIN,
                align="C",
            )
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Página {self.page_no()}", border=0, align="C")

    def chapter_title(self, title):
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, title, new_x=XPos.LMARGIN, border="B")
        self.ln(10)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, title, new_x=XPos.LMARGIN)
        self.ln(8)

    def body_text_from_file(self, text_file):
        self.set_font("Courier", "", 10)
        caminho_completo = os.path.join(RESULT_DIR, text_file)
        if os.path.exists(caminho_completo):
            with open(caminho_completo, "r", encoding="utf-8") as f:
                text = sanitize_text(f.read())
            self.multi_cell(0, 5, text)
        else:
            self.set_font("Helvetica", "I", 10)
            self.multi_cell(0, 5, f"ERRO: Arquivo '{text_file}' nao encontrado.")
        self.ln(5)

    def add_image(self, image_file, width=170):
        caminho_completo = os.path.join(RESULT_DIR, image_file)
        if os.path.exists(caminho_completo):
            self.image(caminho_completo, w=width)
        else:
            self.set_font("Helvetica", "I", 10)
            self.multi_cell(0, 5, f"ERRO: Imagem '{image_file}' nao encontrada.")
        self.ln(5)


def gerar_pdf():
    """
    Gera o relatório final em PDF com todos os resultados.
    """
    print("\n--- Iniciando Geração do Relatório PDF ---")

    # Checagem de segurança
    if not os.path.exists(RESULT_DIR):
        print(f"ERRO: O diretório '{RESULT_DIR}' não foi encontrado.")
        print("Por favor, execute o script 'resolver_problema.py' primeiro.")
        return

    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Informações do Aluno
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Aluno: {sanitize_text(NOME_ALUNO)}", new_x=XPos.LMARGIN)
    pdf.ln(8)
    pdf.cell(0, 8, "Disciplina: Inteligência Artificial", new_x=XPos.LMARGIN)
    pdf.ln(15)

    # --- Exercício 1 ---
    pdf.chapter_title("Exercício 1: Decisão de Aceitar Estágio")
    pdf.section_title("a) Árvore de Decisão Gerada:")
    pdf.add_image("ex1_arvore.png", width=160)
    pdf.section_title("b) Regras SE-ENTÃO Extraídas:")
    pdf.body_text_from_file("ex1_regras.txt")
    pdf.section_title("c) e d) Classificações de Novos Casos:")
    pdf.body_text_from_file("ex1_previsoes.txt")

    # --- Exercício 2 ---
    pdf.add_page()
    pdf.chapter_title("Exercício 2: Decisão de Jogar Tênis")
    pdf.section_title("a) Árvore de Decisão Gerada:")
    pdf.add_image("ex2_arvore.png", width=180)  # Imagem maior
    pdf.section_title("b) Regras SE-ENTÃO Extraídas:")
    pdf.body_text_from_file("ex2_regras.txt")
    pdf.section_title("c) e d) Classificações de Novos Casos:")
    pdf.body_text_from_file("ex2_previsoes.txt")

    # --- Código Fonte ---
    pdf.add_page()
    pdf.chapter_title(f"Anexo: Código Fonte ({NOME_ARQUIVO_SCRIPT})")
    pdf.set_font("Courier", "", 8)
    try:
        with open(NOME_ARQUIVO_SCRIPT, "r", encoding="utf-8") as f:
            code = sanitize_text(f.read())
        pdf.multi_cell(0, 4.5, code)
    except FileNotFoundError:
        pdf.set_font("Helvetica", "I", 10)
        pdf.multi_cell(
            0,
            5,
            f"ERRO: Nao foi possivel encontrar o arquivo de codigo-fonte '{NOME_ARQUIVO_SCRIPT}'.",
        )

    # --- Salvar o PDF ---
    nome_arquivo_pdf = "Relatorio_Primeiro_Trabalho_Arvore_Decisao.pdf"
    pdf.output(nome_arquivo_pdf)
    print(f"\nPDF '{nome_arquivo_pdf}' gerado com sucesso!")


# --- EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    if NOME_ALUNO == "Seu Nome Completo Aqui":
        print(
            "\n!!! ATENÇÃO: A variável NOME_ALUNO não foi encontrada no arquivo .env ou o arquivo não existe. !!!"
        )
        print(
            'Por favor, crie um arquivo .env e adicione a linha: NOME_ALUNO="Seu Nome"'
        )
    else:
        gerar_pdf()
