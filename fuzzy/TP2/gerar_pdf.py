import os
from fpdf import FPDF
from fpdf.enums import XPos
from dotenv import load_dotenv

# --- CONFIGURAÇÕES ---
load_dotenv() 
NOME_AUTOR = os.getenv('NOME', 'Aluno(a)')
NOME_ARQUIVO_SCRIPT = 'resolver_fuzzy.py'
RESULT_DIR = './result'
NOME_ARQUIVO_PDF = 'Relatorio_Trabalho_Fuzzy_2.pdf'

# --- FUNÇÃO PARA SANITIZAR TEXTO ---
def sanitize_text(text):
    """Remove caracteres incompatíveis com a codificação latin-1."""
    return text.encode('latin-1', 'ignore').decode('latin-1')

# --- CLASSE PDF CUSTOMIZADA ---
class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, '2º Trabalho Prático sobre Lógica Fuzzy', 
                  border=0, new_x=XPos.LMARGIN, align='C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', border=0, align='C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, title, new_x=XPos.LMARGIN, border='B')
        self.ln(10)

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 8, title, new_x=XPos.LMARGIN)
        self.ln(6)
    
    def body_text(self, text_file):
        self.set_font('Courier', '', 10)
        with open(os.path.join(RESULT_DIR, text_file), 'r', encoding='utf-8') as f:
            text = sanitize_text(f.read())
        self.multi_cell(0, 5, text)
        self.ln(5)

    def add_image_section(self, title, image_files, w=170):
        self.section_title(title)
        for img_file in image_files:
            path = os.path.join(RESULT_DIR, img_file)
            if os.path.exists(path):
                self.image(path, w=w)
                self.ln(2)
        self.ln(5)

# --- VERIFICAÇÃO DOS ARQUIVOS ---
print("Verificando arquivos de resultado...")
if not os.path.exists(RESULT_DIR):
    print(f"ERRO: Diretório de resultados '{RESULT_DIR}' não encontrado.")
    print(f"Por favor, execute o script '{NOME_ARQUIVO_SCRIPT}' primeiro.")
    exit()
print("Diretório de resultados encontrado.")

# --- GERAÇÃO DO PDF ---
print("Iniciando a geração do PDF...")
pdf = PDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)

pdf.set_font('Helvetica', '', 12)
pdf.cell(0, 8, f'Aluno(a): {NOME_AUTOR}', new_x=XPos.LMARGIN)
pdf.ln(10)

# --- PROBLEMA A ---
pdf.chapter_title('Problema A: Vitalidade das Violetas')
pdf.add_image_section('Funções de Pertinência:', ['exA_agua_mf.png', 'exA_sol_mf.png', 'exA_vitalidade_mf.png'])
pdf.section_title('Base de Regras:')
pdf.body_text('exA_regras.txt')
pdf.section_title('Resultado da Simulação:')
pdf.body_text('exA_resultado.txt')
pdf.add_image_section('Gráfico de Saída (Defuzzificação):', ['exA_vitalidade_resultado_grafico.png'])

# --- PROBLEMA B ---
pdf.add_page()
pdf.chapter_title('Problema B: Modelagem da Política de Crédito')
pdf.add_image_section('Funções de Pertinência:', ['exB_score_mercado_mf.png', 'exB_score_interno_mf.png', 'exB_engajamento_mf.png', 'exB_risco_mf.png'])
pdf.section_title('Base de Regras (Interpretadas):')
pdf.body_text('exB_regras.txt')
pdf.section_title('Resultado da Simulação:')
pdf.body_text('exB_resultado.txt')
pdf.add_image_section('Gráfico de Saída (Defuzzificação):', ['exB_risco_resultado_grafico.png'])

# --- PROBLEMA C ---
pdf.add_page()
pdf.chapter_title('Problema C: Problema da Gorjeta')
pdf.add_image_section('Funções de Pertinência:', ['exC_comida_mf.png', 'exC_servico_mf.png', 'exC_gorjeta_mf.png'])
pdf.section_title('Base de Regras:')
pdf.body_text('exC_regras.txt')
pdf.section_title('Resultado da Simulação:')
pdf.body_text('exC_resultado.txt')
pdf.add_image_section('Gráfico de Saída (Defuzzificação):', ['exC_gorjeta_resultado_grafico.png'])

# --- PROBLEMA D ---
pdf.add_page()
pdf.chapter_title('Problema D: Cálculo do Prêmio de Seguro')
pdf.add_image_section('Funções de Pertinência:', ['exD_idade_mf.png', 'exD_saude_mf.png', 'exD_premio_mf.png'])
pdf.section_title('Base de Regras:')
pdf.body_text('exD_regras.txt')
pdf.section_title('Resultado da Simulação:')
pdf.body_text('exD_resultado.txt')
pdf.add_image_section('Gráfico de Saída (Defuzzificação):', ['exD_premio_resultado_grafico.png'])

# --- CÓDIGO FONTE ---
pdf.add_page()
pdf.ln(5)
pdf.chapter_title(f'Código Fonte: {NOME_ARQUIVO_SCRIPT}')
pdf.set_font('Courier', '', 8)
with open(NOME_ARQUIVO_SCRIPT, 'r', encoding='utf-8') as f:
    code = sanitize_text(f.read())
pdf.multi_cell(0, 4.5, code)

# --- SALVAR O ARQUIVO PDF ---
pdf.output(NOME_ARQUIVO_PDF)

print(f"\nPDF '{NOME_ARQUIVO_PDF}' gerado com sucesso!")