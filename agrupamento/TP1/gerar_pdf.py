# -*- coding: utf-8 -*-

import os
from fpdf import FPDF
from fpdf.enums import XPos
from dotenv import load_dotenv, find_dotenv

# --- CARREGAR VARIÁVEIS DE AMBIENTE ---
load_dotenv(find_dotenv())

# --- CONFIGURAÇÕES ---
NOME_ALUNO = os.getenv("NOME", "Seu Nome Completo Aqui")
GITHUB_PAGES_BASE_URL = os.getenv("GITHUB_PAGES_BASE_URL")
RESULT_DIR = "./result"
NOME_ARQUIVO_SCRIPT = "resolver_trabalho.py"


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
                "Primeiro Trabalho Prático sobre Agrupamento",
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
        """Lê e insere o conteúdo de um arquivo .txt"""
        self.set_font("Courier", "", 8)
        caminho_completo = os.path.join(RESULT_DIR, text_file)

        if os.path.exists(caminho_completo):
            with open(caminho_completo, "r", encoding="utf-8") as f:
                text = sanitize_text(f.read())

            effective_width = self.w - self.l_margin - self.r_margin
            self.set_x(self.l_margin)
            self.multi_cell(effective_width, 5, text)

        else:
            self.set_font("Helvetica", "I", 10)
            self.multi_cell(0, 5, f"ERRO: Arquivo '{text_file}' nao encontrado.")
        self.ln(5)

    def add_image(self, image_file, width=170):
        """Adiciona uma imagem estática (PNG, JPG)"""
        caminho_completo = os.path.join(RESULT_DIR, image_file)
        if os.path.exists(caminho_completo):
            x_pos = (self.w - width) / 2
            # Adiciona quebra de página automática se a imagem não couber
            self.image(caminho_completo, w=width, x=x_pos, keep_aspect_ratio=True)
        else:
            self.set_font("Helvetica", "I", 10)
            self.multi_cell(0, 5, f"ERRO: Imagem '{image_file}' nao encontrada.")
        self.ln(5)

    def add_interactive_image(self, image_file, link_file, width=170):
        """Adiciona a imagem estática E um link para o HTML interativo"""
        self.add_image(image_file, width)
        self.set_font("Helvetica", "U", 10)
        self.set_text_color(0, 0, 255)  # Define a cor azul para o link

        link_filename = os.path.basename(link_file)

        if GITHUB_PAGES_BASE_URL:
            dotenv_path = find_dotenv()
            repo_root = (
                os.path.dirname(os.path.abspath(dotenv_path))
                if dotenv_path
                else os.getcwd()
            )
            script_dir = os.path.abspath(os.path.dirname(__file__))
            relative_script_dir = os.path.relpath(script_dir, repo_root)

            if relative_script_dir == ".":
                relative_path_for_url = "result"
            else:
                relative_path_for_url = os.path.join(
                    relative_script_dir, "result"
                ).replace(os.sep, "/")

            link_url = (
                f"{GITHUB_PAGES_BASE_URL}/{relative_path_for_url}/{link_filename}"
            )

        else:
            link_path = os.path.join(RESULT_DIR, link_filename)
            link_url = f"file:///{os.path.abspath(link_path)}"

        self.cell(
            0,
            5,
            f"Abrir versao interativa ({link_filename})",
            new_x=XPos.LMARGIN,
            link=link_url,
            align="C",
        )
        self.set_text_color(0, 0, 0)  # Reseta a cor do texto para preto
        self.ln(10)


def gerar_pdf():
    """
    Gera o relatório final em PDF com todos os resultados.
    """
    print("\n--- Iniciando Geração do Relatório PDF ---")

    if not os.path.exists(RESULT_DIR):
        print(f"ERRO: O diretório '{RESULT_DIR}' não foi encontrado.")
        print(f"Ele deve existir no mesmo local que o script 'gerar_pdf.py'.")
        print(f"Por favor, execute o script '{NOME_ARQUIVO_SCRIPT}' primeiro.")
        return

    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Informações do Aluno
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Aluno: {sanitize_text(NOME_ALUNO)}")
    pdf.ln(8)
    pdf.cell(0, 8, "Disciplina: Inteligência Artificial")
    pdf.ln(15)

    # --- Problema 1 ---
    pdf.chapter_title("Problema 1: Agrupamento01.txt")

    pdf.section_title("Metodo do Cotovelo (Elbow Method)")
    pdf.add_image("p1_grafico_cotovelo.png", width=150)

    pdf.section_title("Centros Encontrados (K-Means k=5)")
    pdf.body_text_from_file("p1_centros.txt")

    pdf.section_title("Grafico de Resultado (K-Means k=5)")
    pdf.add_image("p1_kmeans_resultado.png", width=150)

    # --- Problema 2 ---
    pdf.add_page()
    pdf.chapter_title("Problema 2: Agrupamento02.txt")

    pdf.section_title("Metodo do Cotovelo (Elbow Method)")
    pdf.add_image("p2_grafico_cotovelo.png", width=150)

    algoritmos = [
        ("a) K-Means", "p2_kmeans", True),
        ("b) Mini Batch K-Means", "p2_minibatch_kmeans", False),
        ("c) DBSCAN", "p2_dbscan", False),
        ("d) Affinity Propagation", "p2_affinity", False),
        ("e) BIRCH", "p2_birch", False),
        ("f) Agglomerative Clustering", "p2_agglomerative", False),
    ]

    MAX_PAGE_HEIGHT = 297 - 15

    for titulo, basename, tem_centros in algoritmos:

        # Estima a altura necessária para o próximo bloco
        # Título(16) + Imagem(aprox 100) + Link(10) + Margem(10)
        estimated_height = 136
        if tem_centros and basename == "p2_kmeans":
            estimated_height += 60  # Espaço extra para os centros do K-Means

        # Verifica se o bloco cabe no espaço restante
        if (pdf.get_y() + estimated_height) > MAX_PAGE_HEIGHT:
            pdf.add_page()  # Pula a página ANTES de desenhar

        # Agora desenha o bloco, sabendo que ele cabe
        pdf.section_title(titulo)

        if tem_centros:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 5, "Centros Encontrados:", new_x=XPos.LMARGIN, align="L")
            pdf.ln(8)
            pdf.body_text_from_file("p2_centros_kmeans.txt")

        pdf.add_interactive_image(f"{basename}.png", f"{basename}.html", width=160)

    # --- Código Fonte ---
    pdf.add_page()
    pdf.chapter_title(f"Anexo: Codigo Fonte ({NOME_ARQUIVO_SCRIPT})")
    pdf.set_font("Courier", "", 8)
    try:
        with open(NOME_ARQUIVO_SCRIPT, "r", encoding="utf-8") as f:
            code = sanitize_text(f.read())

        effective_width = pdf.w - pdf.l_margin - pdf.r_margin
        pdf.multi_cell(effective_width, 4.5, code)

    except FileNotFoundError:
        pdf.set_font("Helvetica", "I", 10)
        pdf.multi_cell(
            0,
            5,
            f"ERRO: Nao foi possivel encontrar o arquivo de codigo-fonte '{NOME_ARQUIVO_SCRIPT}'.",
        )

    # --- Salvar o PDF ---
    nome_arquivo_pdf = "Relatorio_Primeiro_Trabalho_Agrupamento.pdf"
    pdf.output(nome_arquivo_pdf)
    print(f"\nPDF '{nome_arquivo_pdf}' gerado com sucesso!")


# --- EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    script_parent_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_parent_dir)

    if NOME_ALUNO == "Seu Nome Completo Aqui":
        print(
            "\n!!! ATENÇÃO: A variável NOME_ALUNO não foi encontrada no arquivo .env ou o arquivo não existe. !!!"
        )
        print(
            'Por favor, crie um arquivo .env na raiz do projeto e adicione a linha: NOME="Seu Nome Completo"'
        )
    else:
        gerar_pdf()
