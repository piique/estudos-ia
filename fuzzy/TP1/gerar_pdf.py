import os
from fpdf import FPDF
from fpdf.enums import XPos
from dotenv import load_dotenv

# --- CONFIGURAÇÕES ---
load_dotenv() 
NOME_AUTOR = os.getenv('NOME', 'Autor Desconhecido')
NOME_ARQUIVO_SCRIPT = 'resolver_fuzzy.py'
RESULT_DIR = './result'

# --- FUNÇÃO PARA SANITIZAR TEXTO ---
def sanitize_text(text):
    """Remove caracteres incompatíveis com a codificação latin-1."""
    return text.encode('latin-1', 'ignore').decode('latin-1')

# --- CLASSE PDF CUSTOMIZADA ---
class PDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            self.set_font('Helvetica', 'B', 20)
            self.cell(0, 10, 'Relatório do Trabalho Prático de IA - Lógica Fuzzy', 
                      border=0, new_x=XPos.LMARGIN, align='C')
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', border=0, align='C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, title, new_x=XPos.LMARGIN, border='B')
        self.ln(10)

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 8, title, new_x=XPos.LMARGIN)
        self.ln(8)
    
    def body_text(self, text_file):
        self.set_font('Courier', '', 10)
        with open(os.path.join(RESULT_DIR, text_file), 'r', encoding='utf-8') as f:
            text = sanitize_text(f.read())
        self.multi_cell(0, 5, text)
        self.ln(5)

    def add_image_section(self, title, image_files):
        self.section_title(title)
        for img_file in image_files:
            path = os.path.join(RESULT_DIR, img_file)
            if os.path.exists(path):
                self.image(path, w=170)
                self.ln(2)
        self.ln(5)

# --- LEITURA E VERIFICAÇÃO DOS ARQUIVOS ---
print("Verificando arquivos de resultado...")
required_files = [
    'ex1_dinheiro_mf.png', 'ex1_pessoal_mf.png', 'ex1_risco_mf.png', 'ex1_regras.txt', 
    'ex1_resultado.txt', 'ex1_risco_resultado_grafico.png',
    'ex2_experiencia_mf.png', 'ex2_capacitacao_mf.png', 'ex2_gratificacao_mf.png',
    'ex2_regras.txt', 'ex2_resultados.txt',
    'ex3_distancia_mf.png', 'ex3_municao_mf.png', 'ex3_desejabilidade_mf.png',
    'ex3_regras_tabelas.txt', 'ex3_resultados.txt'
]
missing_files = [f for f in required_files if not os.path.exists(os.path.join(RESULT_DIR, f))]
if missing_files:
    print(f"\nERRO: Arquivos não encontrados: {', '.join(missing_files)}")
    print("Por favor, execute o script 'fuzzy_logic_solver.py' primeiro.")
    exit()
print("Todos os arquivos necessários foram encontrados.")

# --- GERAÇÃO DO PDF ---
print("Iniciando a geração do PDF...")
pdf = PDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)

# Informações do autor
pdf.set_font('Helvetica', '', 12)
pdf.cell(0, 10, f'Autor: {NOME_AUTOR}', new_x=XPos.LMARGIN)
pdf.ln(10)

# --- EXERCÍCIO 1 ---
pdf.chapter_title('Exercício 1: Análise de Risco de Projeto')
pdf.add_image_section('Funções de Pertinência:', ['ex1_dinheiro_mf.png', 'ex1_pessoal_mf.png', 'ex1_risco_mf.png'])
pdf.section_title('Base de Conhecimento (Regras):')
pdf.body_text('ex1_regras.txt')
pdf.section_title('Resultado da Simulação:')
pdf.body_text('ex1_resultado.txt')
pdf.add_image_section('Gráfico de Saída (Defuzzificação):', ['ex1_risco_resultado_grafico.png'])

# --- EXERCÍCIO 2 ---
pdf.add_page()
pdf.chapter_title('Exercício 2: Sistema de Gratificação de RH')
pdf.add_image_section('Funções de Pertinência:', ['ex2_experiencia_mf.png', 'ex2_capacitacao_mf.png', 'ex2_gratificacao_mf.png'])
pdf.section_title('Resultados para os Cenários Propostos:')
pdf.body_text('ex2_resultados.txt')

# --- EXERCÍCIO 3 ---
pdf.add_page()
pdf.chapter_title('Exercício 3: IA para Seleção de Armas em Jogo')
pdf.add_image_section('Funções de Pertinência:', ['ex3_distancia_mf.png', 'ex3_municao_mf.png', 'ex3_desejabilidade_mf.png'])
pdf.section_title('Tabelas de Regras (Lógica Definida):')
pdf.body_text('ex3_regras_tabelas.txt')
pdf.section_title('Análise e Resultados dos Cenários:')
pdf.body_text('ex3_resultados.txt')

# --- CÓDIGO FONTE ---
pdf.add_page()
pdf.ln(5)
pdf.chapter_title(f'Código Fonte: {NOME_ARQUIVO_SCRIPT}')
pdf.set_font('Courier', '', 8)
with open(NOME_ARQUIVO_SCRIPT, 'r', encoding='utf-8') as f:
    code = sanitize_text(f.read())
pdf.multi_cell(0, 4.5, code)

# --- SALVAR O ARQUIVO PDF ---
NOME_ARQUIVO_PDF = 'Relatorio_Trabalho_Fuzzy.pdf'
pdf.output(NOME_ARQUIVO_PDF)

print(f"\nPDF '{NOME_ARQUIVO_PDF}' gerado com sucesso!")