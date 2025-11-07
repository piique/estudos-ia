# -*- coding: utf-8 -*-

import os
from fpdf import FPDF
from fpdf.enums import XPos
from dotenv import load_dotenv, find_dotenv

# --- CARREGAR VARIÁVEIS DE AMBIENTE ---
# (Certifique-se que o .env está na pasta TP3 ou na raiz do projeto)
load_dotenv(find_dotenv())

# --- CONFIGURAÇÕES ---
NOME_ALUNO = os.getenv("NOME", "Seu Nome Completo Aqui")
GITHUB_PAGES_BASE_URL = os.getenv("GITHUB_PAGES_BASE_URL")
RESULT_DIR = "./result"
NOME_ARQUIVO_SCRIPT = "resolver_trabalho.py"
NOME_ARQUIVO_PDF = "Relatorio_Terceiro_Trabalho_Agrupamento.pdf"

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
    ("HDBSCAN", "hdbscan"), # Adicionado na mesma ordem do script
    ("Ward", "ward"),
    ("Spectral Clustering", "spectral"),
    ("MeanShift", "meanshift"),
]

# --- ALTURAS ESTIMADAS PARA VERIFICAÇÃO DE PÁGINA (CORRIGIDO) ---
TITLE_HEIGHT = 16       # 8mm cell + 8mm ln
LN_AFTER_IMG = 5        # ln(5) após cada imagem
LINK_HEIGHT = 15        # 5mm cell + 10mm ln

# Gráficos 2D (Cotovelo, P2, P3) são figsize=(12, 8) -> aspect=1.5
# No PDF, são inseridos com width=150mm (default)
IMG_WIDTH_2D = 150
IMG_HEIGHT_2D = IMG_WIDTH_2D / 1.5 # 100mm (Cálculo correto)

# Gráficos 3D (P1) são (800, 600) -> aspect=1.333...
# No PDF, são inseridos com width=160mm
IMG_WIDTH_3D = 160
IMG_HEIGHT_3D = IMG_WIDTH_3D / (800/600) # 120mm (Cálculo correto)

# Bloco (Título + Imagem 2D + ln)
BLOCK_HEIGHT_2D = TITLE_HEIGHT + IMG_HEIGHT_2D + LN_AFTER_IMG # 16 + 100 + 5 = 121
# Bloco (Título + Imagem 3D + ln + Link)
BLOCK_HEIGHT_3D = TITLE_HEIGHT + IMG_HEIGHT_3D + LN_AFTER_IMG + LINK_HEIGHT # 16 + 120 + 5 + 15 = 156
# --- FIM DAS ALTURAS ESTIMADAS ---


def sanitize_text(text):
    """Remove caracteres incompatíveis com a codificação latin-1 do FPDF."""
    return text.encode("latin-1", "ignore").decode("latin-1")


class PDF(FPDF):
    def header(self):
        # Título para o TP3
        if self.page_no() == 1:
            self.set_font("Helvetica", "B", 18)
            self.cell(
                0,
                10,
                "Terceiro Trabalho Prático sobre Agrupamento", # Atualizado
                border=0,
                new_x=XPos.LMARGIN,
                align="C",
            )
        self.ln(10)

    def footer(self):
        self.set_y(-15)

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

    def check_block_fits(self, height_estimate):
        """Verifica se um bloco cabe, se não, adiciona nova página."""
        if (self.get_y() + height_estimate) > (self.h - self.b_margin):
            self.add_page()
            
    def add_image(self, image_file, width=150, skip_page_check=False):
        """Adiciona uma imagem estática (PNG, JPG) centralizada"""
        caminho_completo = os.path.join(RESULT_DIR, image_file)
        
        # Verifica também se há uma imagem de ERRO
        caminho_erro = caminho_completo.replace(".png", "_ERRO.png")

        if os.path.exists(caminho_completo):
            if not skip_page_check:
                # Usa a altura 2D como fallback se chamada diretamente
                if (self.get_y() + IMG_HEIGHT_2D) > (self.h - self.b_margin):
                    self.add_page()

            x_pos = (self.w - width) / 2
            self.image(caminho_completo, w=width, x=x_pos, keep_aspect_ratio=True)
            self.ln(LN_AFTER_IMG)  # Adiciona espaço após a imagem
        
        elif os.path.exists(caminho_erro):
            self.set_font("Helvetica", "I", 10)
            self.set_text_color(255, 0, 0)  # Vermelho
            self.multi_cell(0, 5, f"AVISO: A imagem '{image_file}' falhou ao ser gerada. Verifique o console.")
            self.set_text_color(0, 0, 0)  # Preto
            self.ln(5)
            if not skip_page_check:
                if (self.get_y() + IMG_HEIGHT_2D) > (self.h - self.b_margin):
                    self.add_page()
            x_pos = (self.w - width) / 2
            self.image(caminho_erro, w=width, x=x_pos, keep_aspect_ratio=True)
            self.ln(LN_AFTER_IMG)

        else:
            self.set_font("Helvetica", "I", 10)
            self.set_text_color(255, 0, 0)  # Vermelho
            self.multi_cell(0, 5, f"ERRO: Imagem '{image_file}' nao encontrada.")
            self.set_text_color(0, 0, 0)  # Preto
            self.ln(5)

    def add_interactive_image(self, image_file, link_file, width=160, skip_page_check=False):
        """Adiciona a imagem estática E um link para o HTML interativo"""

        if not skip_page_check:
            # Usa a altura 3D como fallback se chamada diretamente
            if (self.get_y() + IMG_HEIGHT_3D + LINK_HEIGHT) > (self.h - self.b_margin):
                self.add_page()

        self.add_image(image_file, width, skip_page_check=True)
        
        caminho_html = os.path.join(RESULT_DIR, link_file)
        if not os.path.exists(caminho_html):
            return

        self.set_font("Helvetica", "U", 10)
        self.set_text_color(0, 0, 255)  # Define a cor azul para o link

        link_filename = os.path.basename(link_file)
        final_link_url = ""

        if GITHUB_PAGES_BASE_URL:
            # Lógica para GITHUB_PAGES
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

            # --- REVERSÃO ---
            # Voltando ao link absoluto simples, sem JavaScript.
            final_link_url = (
                f"{GITHUB_PAGES_BASE_URL}/{relative_path_for_url}/{link_filename}"
            )

        else:
            final_link_url = os.path.join("result", link_filename).replace(os.sep, "/")


        self.cell(
            0,
            5,
            f"Abrir versao interativa ({link_filename})",
            new_x=XPos.LMARGIN,
            link=final_link_url, # Link padrão
            align="C",
        )
        self.set_text_color(0, 0, 0)  # Reseta a cor do texto para preto
        self.ln(10) # 10mm de ln(10)


def gerar_pdf():
    """
    Gera o relatório final em PDF com todos os resultados do TP3.
    """
    print(f"\n--- Iniciando Geração do Relatório PDF ({NOME_ARQUIVO_PDF}) ---")

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

    # --- Problema 1: Agrupamento06.txt (3D) ---
    pdf.chapter_title("Problema 1: Agrupamento06.txt")
    
    # Verifica o bloco 2D (Cotovelo)
    pdf.check_block_fits(BLOCK_HEIGHT_2D)
    pdf.section_title("Metodo do Cotovelo (Elbow Method)")
    pdf.add_image("p1_grafico_cotovelo.png", skip_page_check=True)

    for i, (nome_amigavel, nome_arquivo) in enumerate(ALGORITMOS, 1):
        # Verifica o bloco 3D (Interativo)
        pdf.check_block_fits(BLOCK_HEIGHT_3D)
        pdf.section_title(
            f"1.{i}) {nome_amigavel}"
        )
        pdf.add_interactive_image(f"p1_{nome_arquivo}.png", f"p1_{nome_arquivo}.html", skip_page_check=True)

    # --- Problema 2: Agrupamento07.txt (2D) ---
    pdf.add_page()
    pdf.chapter_title("Problema 2: Agrupamento07.txt")

    # Verifica o bloco 2D (Cotovelo)
    pdf.check_block_fits(BLOCK_HEIGHT_2D)
    pdf.section_title("Metodo do Cotovelo (Elbow Method)")
    pdf.add_image("p2_grafico_cotovelo.png", skip_page_check=True)

    for i, (nome_amigavel, nome_arquivo) in enumerate(ALGORITMOS, 1):
        # Verifica o bloco 2D
        pdf.check_block_fits(BLOCK_HEIGHT_2D)
        pdf.section_title(
            f"2.{i}) {nome_amigavel}"
        )
        pdf.add_image(f"p2_{nome_arquivo}.png", skip_page_check=True)

    # --- Problema 3: Agrupamento08.txt (2D) ---
    pdf.add_page()
    pdf.chapter_title("Problema 3: Agrupamento08.txt")

    # Verifica o bloco 2D (Cotovelo)
    pdf.check_block_fits(BLOCK_HEIGHT_2D)
    pdf.section_title("Metodo do Cotovelo (Elbow Method)")
    pdf.add_image("p3_grafico_cotovelo.png", skip_page_check=True)

    for i, (nome_amigavel, nome_arquivo) in enumerate(ALGORITMOS, 1):
        # Verifica o bloco 2D
        pdf.check_block_fits(BLOCK_HEIGHT_2D)
        pdf.section_title(
            f"3.{i}) {nome_amigavel}"
        )
        pdf.add_image(f"p3_{nome_arquivo}.png", skip_page_check=True)

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
    try:
        pdf.output(NOME_ARQUIVO_PDF)
        print(f"\nPDF '{NOME_ARQUIVO_PDF}' gerado com sucesso!")
    except Exception as e:
        print(f"\nERRO ao salvar o PDF: {e}")
        print("Verifique se o arquivo PDF não está aberto em outro programa.")


# --- EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    # Garante que o script rode a partir do diretório onde ele está
    script_parent_dir = os.path.dirname(os.path.abspath(__file__))
    if script_parent_dir:
        os.chdir(script_parent_dir)
        print(f"Trabalhando no diretório: {os.getcwd()}")


    if NOME_ALUNO == "Seu Nome Completo Aqui":
        print(
            "\n!!! ATENÇÃO: A variável NOME_ALUNO não foi encontrada no arquivo .env ou o arquivo não existe. !!!"
        )
        print(
            f'Por favor, crie um arquivo .env na pasta {os.path.basename(os.getcwd())} e adicione a linha: NOME="Seu Nome Completo"'
        )
    else:
        print(f"Gerando relatório para: {sanitize_text(NOME_ALUNO)}")

    gerar_pdf()