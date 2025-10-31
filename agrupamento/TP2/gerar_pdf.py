# -*- coding: utf-8 -*-

import os
from fpdf import FPDF
from fpdf.enums import XPos
from dotenv import load_dotenv, find_dotenv

# --- CARREGAR VARIÁVEIS DE AMBIENTE ---
# (Certifique-se que o .env está na pasta TP2 ou na raiz do projeto)
load_dotenv(find_dotenv())

# --- CONFIGURAÇÕES ---
NOME_ALUNO = os.getenv("NOME", "Seu Nome Completo Aqui")
GITHUB_PAGES_BASE_URL = os.getenv("GITHUB_PAGES_BASE_URL")
RESULT_DIR = "./result"
NOME_ARQUIVO_SCRIPT = "resolver_trabalho.py"
NOME_ARQUIVO_PDF = "Relatorio_Segundo_Trabalho_Agrupamento.pdf"

# Lista dos 12 algoritmos (Nome Amigável, nome_base_arquivo)
ALGORITMOS = [
    ("K-Means", "kmeans"),
    ("Mini Batch K-Means", "minibatch_kmeans"),
    ("DBSCAN", "dbscan"),
    ("Affinity Propagation", "affinity"),
    ("BIRCH", "birch"),
    ("Agglomerative Clustering", "agglomerative"),
    ("Gaussian Mixture", "gmm"),
    ("OPTICS", "optics"),
    ("HDBSCAN", "hdbscan"),
    ("Ward", "ward"),
    ("Spectral Clustering", "spectral"),
    ("MeanShift", "meanshift"),
]


def sanitize_text(text):
    """Remove caracteres incompatíveis com a codificação latin-1 do FPDF."""
    return text.encode("latin-1", "ignore").decode("latin-1")


class PDF(FPDF):
    def header(self):
        # Título diferente para o TP2
        self.set_font("Helvetica", "B", 18)
        self.cell(
            0,
            10,
            "Segundo Trabalho Prático sobre Agrupamento",
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

    def body_text(self, text):
        """Insere um texto simples no corpo"""
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5, sanitize_text(text))
        self.ln(5)

    def add_image(self, image_file, width=150):
        """Adiciona uma imagem estática (PNG, JPG) centralizada"""
        caminho_completo = os.path.join(RESULT_DIR, image_file)
        if os.path.exists(caminho_completo):
            # Verifica se a imagem cabe, senão adiciona página
            if (self.get_y() + 80) > (self.h - self.b_margin):  # Estima 80mm de altura
                self.add_page()

            x_pos = (self.w - width) / 2
            self.image(caminho_completo, w=width, x=x_pos, keep_aspect_ratio=True)
            self.ln(5)  # Adiciona espaço após a imagem
        else:
            self.set_font("Helvetica", "I", 10)
            self.set_text_color(255, 0, 0)  # Vermelho
            self.multi_cell(0, 5, f"ERRO: Imagem '{image_file}' nao encontrada.")
            self.set_text_color(0, 0, 0)  # Preto
            self.ln(5)

    def add_interactive_image(self, image_file, link_file, width=160):
        """Adiciona a imagem estática E um link para o HTML interativo"""

        # Verifica se o bloco cabe
        if (self.get_y() + 100) > (self.h - self.b_margin):  # Estima 100mm
            self.add_page()

        self.add_image(image_file, width)
        self.set_font("Helvetica", "U", 10)
        self.set_text_color(0, 0, 255)  # Define a cor azul para o link

        link_filename = os.path.basename(link_file)

        if GITHUB_PAGES_BASE_URL:
            # Lógica para GITHUB_PAGES (copiada do TP1)
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
            # Cria um link relativo para funcionar melhor localmente
            link_url = os.path.join("result", link_filename)

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
    Gera o relatório final em PDF com todos os resultados do TP2.
    """
    print("\n--- Iniciando Geração do Relatório PDF (TP2) ---")

    if not os.path.exists(RESULT_DIR):
        print(f"ERRO: O diretório '{RESULT_DIR}' não foi encontrado.")
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

    # --- Problema 1: Agrupamento03.txt (2D) ---
    pdf.chapter_title("Problema 1: Agrupamento03.txt")
    pdf.section_title("Metodo do Cotovelo (Elbow Method)")
    pdf.add_image("p1_grafico_cotovelo.png")

    for nome_amigavel, nome_arquivo in ALGORITMOS:
        pdf.section_title(
            f"1.{ALGORITMOS.index((nome_amigavel, nome_arquivo)) + 1}) {nome_amigavel}"
        )
        pdf.add_image(f"p1_{nome_arquivo}.png")

    # --- Problema 2: iris_cluster.txt (4D -> 2D PCA) ---
    pdf.add_page()
    pdf.chapter_title("Problema 2: iris_cluster.txt")
    pdf.section_title("Metodo do Cotovelo (Elbow Method)")
    pdf.add_image("p2_grafico_cotovelo.png")
    pdf.body_text(
        "Nota: Os gráficos a seguir são plotados usando os 2 primeiros "
        "Componentes Principais (PCA) para visualização 2D."
    )

    for nome_amigavel, nome_arquivo in ALGORITMOS:
        pdf.section_title(
            f"2.{ALGORITMOS.index((nome_amigavel, nome_arquivo)) + 1}) {nome_amigavel}"
        )
        pdf.add_image(f"p2_{nome_arquivo}.png")

    # --- Problema 3: Agrupamento04.txt (3D) ---
    pdf.add_page()
    pdf.chapter_title("Problema 3: Agrupamento04.txt")
    pdf.section_title("Metodo do Cotovelo (Elbow Method)")
    pdf.add_image("p3_grafico_cotovelo.png")

    for nome_amigavel, nome_arquivo in ALGORITMOS:
        pdf.section_title(
            f"3.{ALGORITMOS.index((nome_amigavel, nome_arquivo)) + 1}) {nome_amigavel}"
        )
        # Usamos a função interativa para o 3D
        pdf.add_interactive_image(f"p3_{nome_arquivo}.png", f"p3_{nome_arquivo}.html")

    # --- Problema 4: Agrupamento05.txt (2D) ---
    pdf.add_page()
    pdf.chapter_title("Problema 4: Agrupamento05.txt")
    pdf.section_title("Metodo do Cotovelo (Elbow Method)")
    pdf.add_image("p4_grafico_cotovelo.png")

    for nome_amigavel, nome_arquivo in ALGORITMOS:
        pdf.section_title(
            f"4.{ALGORITMOS.index((nome_amigavel, nome_arquivo)) + 1}) {nome_amigavel}"
        )
        pdf.add_image(f"p4_{nome_arquivo}.png")

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
    pdf.output(NOME_ARQUIVO_PDF)
    print(f"\nPDF '{NOME_ARQUIVO_PDF}' gerado com sucesso!")


# --- EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    # Garante que o script rode a partir do diretório onde ele está
    script_parent_dir = os.path.dirname(os.path.abspath(__file__))
    if script_parent_dir:
        os.chdir(script_parent_dir)

    if NOME_ALUNO == "Seu Nome Completo Aqui":
        print(
            "\n!!! ATENÇÃO: A variável NOME_ALUNO não foi encontrada no arquivo .env ou o arquivo não existe. !!!"
        )
        print(
            'Por favor, crie um arquivo .env na pasta TP2 e adicione a linha: NOME="Seu Nome Completo"'
        )
    else:
        print(f"Gerando relatório para: {sanitize_text(NOME_ALUNO)}")

    gerar_pdf()
