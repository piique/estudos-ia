import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()
# Pega o nome do autor do .env, com um valor padrão caso não encontre
NOME_AUTOR = os.getenv('NOME', 'Autor Desconhecido')

# --- [VERSÃO FINAL] FUNÇÃO PARA LIMPAR O TEXTO DE FORMA INTELIGENTE ---
def sanitize_text(text):
    """
    Substitui caracteres de desenho de tabela Unicode por equivalentes ASCII
    e remove silenciosamente quaisquer outros caracteres incompatíveis com latin-1.
    """
    # Dicionário de substituição mais completo para o Keras model.summary()
    replacements = {
        '═': '=', '─': '-', '│': '|', '┌': '+', '┐': '+',
        '└': '+', '┘': '+', '┴': '+', '┬': '+', '├': '+', '┤': '+'
    }
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)
    
    # Rede de segurança: remove quaisquer outros caracteres incompatíveis
    # em vez de substituí-los por '?', usando 'ignore'.
    return text.encode('latin-1', 'ignore').decode('latin-1')

# --- LEITURA DOS DADOS DOS ARQUIVOS ---
print("Lendo os dados dos arquivos gerados...")

try:
    with open('./result/resumo_modelo.txt', 'r', encoding='utf-8') as f:
        resumo_modelo_texto = sanitize_text(f.read())

    with open('./result/legenda_classes.txt', 'r', encoding='utf-8') as f:
        legenda_classes_texto = sanitize_text(f.read())
        
    with open('./result/acuracia.txt', 'r', encoding='utf-8') as f:
        acuracia_final_texto = f.read()

    previsoes_df = pd.read_csv('./result/previsoes_finais.csv')

    with open('treinamento.py', 'r', encoding='utf-8') as f:
        codigo_fonte_texto = sanitize_text(f.read())

except FileNotFoundError as e:
    print(f"Erro: Arquivo não encontrado - {e}")
    print("Por favor, execute o script 'treinamento.py' primeiro e verifique os nomes/caminhos dos arquivos.")
    exit()


# --- GERAÇÃO DO PDF ---
print("Iniciando a geração do PDF...")

class PDF(FPDF):
    def header(self):
        if(self.page_no() == 1):
            self.set_font('Helvetica', 'B', 12)
            self.cell(0, 10, 'Relatório de Treinamento e Predição de Rede Neural', 
                    border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', border=0, align='C')

pdf = PDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)

# --- INFORMAÇÕES DO AUTOR E TÍTULO ---
pdf.set_font('Helvetica', '', 12)
pdf.cell(0, 10, f'Autor: {NOME_AUTOR}', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(10)

# --- RESUMO DO MODELO ---
pdf.set_font('Helvetica', 'B', 14)
pdf.cell(0, 10, '1. Resumo do Modelo', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Courier', '', 10)
pdf.multi_cell(0, 5, resumo_modelo_texto)
pdf.ln(5)

# --- SEÇÃO DE ACURÁCIA ---
pdf.set_font('Helvetica', 'B', 14)
pdf.cell(0, 10, '2. Desempenho do Treinamento', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Helvetica', '', 12)
pdf.cell(0, 8, f"Acurácia final alcançada: {acuracia_final_texto}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(5)

# --- GRÁFICO DO HISTÓRICO DE TREINAMENTO ---
pdf.set_font('Helvetica', 'B', 14)
pdf.cell(0, 10, '3. Gráfico do Histórico de Treinamento', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.image('./result/Historico_de_treinamento.png', x=None, y=None, w=180)
pdf.ln(5)

# --- RESULTADOS DA PREDIÇÃO ---
pdf.add_page()
pdf.set_font('Helvetica', 'B', 14)
pdf.cell(0, 10, '4. Resultados da Predição', new_x=XPos.LMARGIN, new_y=YPos.NEXT)

# Legenda das Classes
pdf.set_font('Helvetica', 'B', 12)
pdf.cell(0, 10, 'Legenda das Classes:', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Courier', '', 10)
pdf.multi_cell(0, 5, legenda_classes_texto)
pdf.ln(5)

# Resposta Final (DataFrame)
pdf.set_font('Helvetica', 'B', 12)
pdf.cell(0, 10, 'Previsões Finais:', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Courier', '', 10)
previsoes_string = previsoes_df.to_string()
pdf.multi_cell(0, 5, sanitize_text(previsoes_string))
pdf.ln(5)

# --- SEÇÃO DO CÓDIGO FONTE ---
pdf.add_page()
pdf.set_font('Helvetica', 'B', 14)
pdf.cell(0, 10, '5. Código Fonte (treinamento.py)', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Courier', '', 8) # Fonte menor para o código caber melhor
pdf.multi_cell(0, 5, codigo_fonte_texto)

# --- SALVAR O ARQUIVO PDF ---
NOME_ARQUIVO_PDF = 'relatorio_treinamento_npc.pdf'
pdf.output(NOME_ARQUIVO_PDF)

print(f"PDF '{NOME_ARQUIVO_PDF}' gerado com sucesso!")