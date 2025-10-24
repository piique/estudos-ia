import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os
from dotenv import load_dotenv

# --- CONFIGURAÇÕES ---
# Carrega as variáveis de ambiente do arquivo .env
load_dotenv() 
# Pega o nome do autor do .env, com um valor padrão caso não encontre
NOME_AUTOR = os.getenv('NOME', 'Autor Desconhecido')
# [AJUSTE] Nome do arquivo do script de treinamento para ser incluído no PDF
NOME_ARQUIVO_TREINAMENTO = 'treinamento.py' 

# --- FUNÇÃO PARA SANITIZAR TEXTO ---
def sanitize_text(text):
    """
    Substitui caracteres de desenho de tabela Unicode por equivalentes ASCII
    e remove silenciosamente quaisquer outros caracteres incompatíveis com latin-1.
    """
    replacements = {
        '═': '=', '─': '-', '│': '|', '┌': '+', '┐': '+',
        '└': '+', '┘': '+', '┴': '+', '┬': '+', '├': '+', '┤': '+'
    }
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)
    return text.encode('latin-1', 'ignore').decode('latin-1')

# --- LEITURA DOS DADOS DOS ARQUIVOS ---
print("Lendo os dados dos arquivos gerados pelo script de treinamento...")

try:
    with open('./result/resumo_modelo.txt', 'r', encoding='utf-8') as f:
        resumo_modelo_texto = sanitize_text(f.read())

    with open('./result/detalhes_rede.txt', 'r', encoding='utf-8') as f:
        detalhes_rede_texto = sanitize_text(f.read())

    with open('./result/legenda_classes.txt', 'r', encoding='utf-8') as f:
        legenda_classes_texto = sanitize_text(f.read())
        
    with open('./result/acuracia_teste.txt', 'r', encoding='utf-8') as f:
        acuracia_final_texto = f.read()

    previsoes_df = pd.read_csv('./result/previsoes_finais.csv')

    with open(NOME_ARQUIVO_TREINAMENTO, 'r', encoding='utf-8') as f:
        codigo_fonte_texto = sanitize_text(f.read())

except FileNotFoundError as e:
    print(f"\nERRO: Arquivo não encontrado - {e.filename}")
    print("Por favor, execute o script de treinamento primeiro e verifique se todos os arquivos foram gerados na pasta './result'.")
    exit()

# --- GERAÇÃO DO PDF ---
print("Iniciando a geração do PDF...")

class PDF(FPDF):
    def header(self):
        # O cabeçalho só aparece na primeira página
        if self.page_no() == 1:
            self.set_font('Helvetica', 'B', 14)
            self.cell(0, 10, 'Relatório de Treinamento de Rede Neural Convolucional (CNN) para identificar Roupas em Imagens', 
                      border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', border=0, align='C')

pdf = PDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)

# --- INFORMAÇÕES DO AUTOR ---
pdf.set_font('Helvetica', '', 12)
pdf.cell(0, 10, f'Autor: {NOME_AUTOR}', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(10)

# --- 1. RESUMO DO MODELO ---
pdf.set_font('Helvetica', 'B', 14)
pdf.cell(0, 10, '1. Resumo do Modelo (Arquitetura)', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Courier', '', 10)
pdf.multi_cell(0, 5, resumo_modelo_texto)
pdf.ln(5)

# --- [NOVO] 2. DETALHES DA REDE ---
pdf.set_font('Helvetica', 'B', 14)
pdf.cell(0, 10, '2. Detalhes da Rede (Pesos e Bias)', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Courier', '', 10)
pdf.multi_cell(0, 5, detalhes_rede_texto)
pdf.ln(5)

# --- 3. DESEMPENHO DO TREINAMENTO ---
pdf.set_font('Helvetica', 'B', 14)
pdf.cell(0, 10, '3. Desempenho do Treinamento', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Helvetica', '', 12)
pdf.cell(0, 8, f"Acurácia final na base de testes: {acuracia_final_texto}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(5)

# --- 4. GRÁFICO DO HISTÓRICO DE TREINAMENTO ---
pdf.set_font('Helvetica', 'B', 14)
pdf.cell(0, 10, '4. Gráfico do Histórico de Treinamento', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
if os.path.exists('./result/Historico_de_treinamento.png'):
    pdf.image('./result/Historico_de_treinamento.png', x=None, y=None, w=180)
else:
    pdf.set_font('Helvetica', 'I', 10)
    pdf.cell(0, 10, 'Imagem do gráfico não encontrada.', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(5)

# --- 5. RESULTADOS DA PREDIÇÃO ---
pdf.add_page()
pdf.set_font('Helvetica', 'B', 14)
pdf.cell(0, 10, '5. Resultados da Predição na Base de Testes', new_x=XPos.LMARGIN, new_y=YPos.NEXT)

# Legenda das Classes
pdf.set_font('Helvetica', 'B', 12)
pdf.cell(0, 10, 'Legenda das Classes:', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Courier', '', 10)
pdf.multi_cell(0, 5, legenda_classes_texto)
pdf.ln(5)

# Tabela de Previsões
pdf.set_font('Helvetica', 'B', 12)
pdf.cell(0, 10, 'Previsões Finais (Amostra):', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Courier', '', 9)
# Mostra apenas as primeiras 30 previsões para não sobrecarregar o PDF
previsoes_string = previsoes_df.head(30).to_string()
pdf.multi_cell(0, 5, sanitize_text(previsoes_string))
pdf.ln(5)

# --- 6. CÓDIGO FONTE ---
pdf.add_page()
pdf.set_font('Helvetica', 'B', 14)
pdf.cell(0, 10, f'6. Código Fonte ({NOME_ARQUIVO_TREINAMENTO})', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Courier', '', 8)
pdf.multi_cell(0, 4.5, codigo_fonte_texto)

# --- SALVAR O ARQUIVO PDF ---
NOME_ARQUIVO_PDF = 'relatorio_treinamento_roupas.pdf'
pdf.output(NOME_ARQUIVO_PDF)

print(f"\nPDF '{NOME_ARQUIVO_PDF}' gerado com sucesso na pasta raiz do projeto!")