# ==============================================================================
# ATENÇÃO: Este script requer a biblioteca fpdf2.
# Para instalar, execute no seu terminal: pip install fpdf2
# ==============================================================================

import io
from contextlib import redirect_stdout
from fpdf import FPDF
from fpdf.enums import XPos, YPos

def solve_and_capture_output():
    """
    Realiza a tradução das sentenças para lógica de predicados e imprime a análise.
    O output é capturado para ser usado no PDF.
    """
    print("Questao 3: Traducao para Logica de Predicados")
    print("=" * 60)
    print("Predicados e Constantes:")
    print("  - PaiOuMae(p, a): p e pai ou mae de a.")
    print("  - Feminino(p): p e do sexo feminino.")
    print("  - Constantes: Joan, Kevin")
    print("  - Quantificador: E!x (existe exatamente um x)")
    print("-" * 60)

    # --- Sentença a) ---
    print("\na) Joan tem uma filha (possivelmente mais de uma e, possivelmente, filhos tambem).")
    print("   Analise:")
    print("   - A sentenca afirma que existe PELO MENOS UMA pessoa 'x' tal que:")
    print("     1. Joan e mae ou pai de 'x' (PaiOuMae(Joan, x)).")
    print("     2. 'x' e do sexo feminino (Feminino(x)).")
    print("   - O quantificador existencial 'E' (existe) e o apropriado aqui.")
    print("\n   Traducao Logica:")
    print("   Ex, (PaiOuMae(Joan, x) ^ Feminino(x))")
    print("-" * 60)

    # --- Sentença b) ---
    print("\nb) Joan tem exatamente uma filha (mas pode ter filhos tambem).")
    print("   Analise:")
    print("   - A sentenca afirma que existe EXATAMENTE UMA pessoa 'x' que e filha de Joan.")
    print("   - Isso significa que existe uma pessoa 'x' que e filha, E qualquer outra")
    print("     pessoa 'y' que tambem seja filha de Joan deve ser, na verdade, a mesma pessoa 'x'.")
    print("   - Usamos o quantificador 'E!x' (existe um e somente um x).")
    print("\n   Traducao Logica (usando E!):")
    print("   E!x, (PaiOuMae(Joan, x) ^ Feminino(x))")
    print("\n   Traducao Logica (forma expandida):")
    print("   Ex, ( (PaiOuMae(Joan, x) ^ Feminino(x)) ^ (Ay, ((PaiOuMae(Joan, y) ^ Feminino(y)) -> y = x)) )")
    print("-" * 60)

    # --- Sentença c) ---
    print("\nc) Joan tem exatamente um filho ou filha.")
    print("   Analise:")
    print("   - Similar a anterior, mas agora a condicao nao especifica o sexo.")
    print("   - A condicao e simplesmente que Joan seja pai ou mae da pessoa 'x'.")
    print("\n   Traducao Logica (usando E!):")
    print("   E!x, PaiOuMae(Joan, x)")
    print("\n   Traducao Logica (forma expandida):")
    print("   Ex, ( PaiOuMae(Joan, x) ^ (Ay, (PaiOuMae(Joan, y) -> y = x)) )")
    print("-" * 60)
    
    # --- Sentença d) ---
    print("\nd) Joan e Kevin tem exatamente um filho ou filha juntos.")
    print("   Analise:")
    print("   - A condicao agora e que a pessoa 'x' tenha tanto Joan QUANTO Kevin como pais/maes.")
    print("   - A afirmacao e que existe exatamente uma pessoa 'x' que satisfaz essa condicao dupla.")
    print("\n   Traducao Logica (usando E!):")
    print("   E!x, (PaiOuMae(Joan, x) ^ PaiOuMae(Kevin, x))")
    print("\n   Traducao Logica (forma expandida):")
    print("   Ex, ( (PaiOuMae(Joan, x) ^ PaiOuMae(Kevin, x)) ^ (Ay, ((PaiOuMae(Joan, y) ^ PaiOuMae(Kevin, y)) -> y = x)) )")
    print("-" * 60)

    # --- Sentença e) ---
    print("\ne) Joan tem pelo menos um filho ou filha com Kevin e nao tem filhos com mais ninguem.")
    print("   Analise:")
    print("   - Esta e uma sentenca composta de duas partes conectadas por 'E':")
    print("     Parte 1: Joan tem pelo menos um filho/filha com Kevin.")
    print("       - Ex, (PaiOuMae(Joan, x) ^ PaiOuMae(Kevin, x))")
    print("     Parte 2: Joan nao tem filhos com mais ninguem.")
    print("       - Isso significa que PARA TODA pessoa 'y', se Joan e pai/mae de 'y',")
    print("         entao Kevin TAMBEM DEVE SER pai/mae de 'y'.")
    print("       - Ay, (PaiOuMae(Joan, y) -> PaiOuMae(Kevin, y))")
    print("\n   Traducao Logica (combinando as duas partes):")
    print("   (Ex, (PaiOuMae(Joan, x) ^ PaiOuMae(Kevin, x))) ^ (Ay, (PaiOuMae(Joan, y) -> PaiOuMae(Kevin, y)))")
    print("=" * 60)


def generate_pdf_report(code_content, output_content):
    """Gera um PDF com o código e o output."""
    pdf = FPDF()
    pdf.add_page()
    
    # Título Principal
    pdf.set_font("Courier", 'B', 16)
    pdf.cell(0, 10, 'Questao 3', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.ln(10)

    # Seção do Código Fonte
    # pdf.set_font("Courier", 'B', 12)
    # pdf.cell(0, 10, 'Codigo Fonte:', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    # pdf.set_font("Courier", '', 8)
    # # Substituindo caracteres unicode que podem causar erro
    # code_content_safe = code_content.encode('latin-1', 'replace').decode('latin-1')
    # pdf.multi_cell(0, 5, code_content_safe)
    
    # pdf.add_page()

    # Seção do Output (Análise Lógica)
    pdf.set_font("Courier", 'B', 12)
    pdf.cell(0, 10, 'Output da Execucao (Analise Logica):', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Courier", '', 10)
    # Substituindo caracteres unicode que podem causar erro
    output_content_safe = output_content.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 5, output_content_safe)

    pdf_file_name = "resultado_questao_3.pdf"
    pdf.output(pdf_file_name)
    return pdf_file_name

# --- Bloco Principal de Execução ---
if __name__ == "__main__":
    # Captura o output da execução
    output_buffer = io.StringIO()
    with redirect_stdout(output_buffer):
        solve_and_capture_output()
    output_content = output_buffer.getvalue()

    # Lê o conteúdo do próprio script
    try:
        with open(__file__, 'r', encoding='utf-8') as f:
            code_content = f.read()
    except Exception as e:
        code_content = f"Nao foi possivel ler o arquivo do codigo: {e}"

    # Gera o PDF
    try:
        pdf_file = generate_pdf_report(code_content, output_content)
        print(f"\nPDF '{pdf_file}' gerado com sucesso no diretorio atual!")
    except Exception as e:
        print(f"\nOcorreu um erro ao gerar o PDF: {e}")