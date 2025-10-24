import io
from contextlib import redirect_stdout
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import warnings
warnings.filterwarnings('ignore')

def solve_and_capture_output():
    """
    Executa a dedução lógica passo a passo para a Questão 7.
    """
    print("Questao 7: Deducao de Valores Verdade")
    print("=" * 70)
    print("Premissas Iniciais:")
    print("1. A proposicao (p -> (r v s)) e FALSA.")
    print("2. A proposicao ((q ^ ~s) <-> p) e VERDADEIRA.")
    print("-" * 70)

    # --- Passo 1: Análise da Premissa 1 ---
    print("\nPasso 1: Analisando a Premissa 1: (p -> (r v s)) e FALSA")
    print("-" * 70)
    print("A unica maneira de uma implicacao (A -> B) ser FALSA e quando")
    print("o antecedente (A) e VERDADEIRO e o consequente (B) e FALSO.")
    print("\n  - Portanto, para (p -> (r v s)) ser FALSA, temos que:")
    print("    -> p = VERDADEIRO")
    print("    -> (r v s) = FALSO")
    
    print("\nAgora, analisando a expressao (r v s) = FALSO:")
    print("A unica maneira de uma disjuncao (OU / v) ser FALSA e quando")
    print("ambas as suas partes sao FALSAS.")
    print("\n  - Portanto, para (r v s) ser FALSO, temos que:")
    print("    -> r = FALSO")
    print("    -> s = FALSO")
    
    # --- Passo 2: Análise da Premissa 2 ---
    print("\n\nPasso 2: Analisando a Premissa 2: ((q ^ ~s) <-> p) e VERDADEIRA")
    print("-" * 70)
    print("Ja sabemos do Passo 1 que p = VERDADEIRO.")
    print("Vamos substituir 'p' na segunda premissa:")
    print("  ((q ^ ~s) <-> VERDADEIRO) e VERDADEIRA")
    
    print("\nUma bi-implicacao (A <-> B) e VERDADEIRA quando ambas as partes")
    print("tem o mesmo valor verdade.")
    print("Como o lado direito e VERDADEIRO, o lado esquerdo tambem deve ser.")
    print("\n  - Portanto, temos que:")
    print("    -> (q ^ ~s) = VERDADEIRO")

    print("\nAgora, analisando a expressao (q ^ ~s) = VERDADEIRO:")
    print("A unica maneira de uma conjuncao (E / ^) ser VERDADEIRA e quando")
    print("ambas as suas partes sao VERDADEIRAS.")
    print("\n  - Portanto, para (q ^ ~s) ser VERDADEIRO, temos que:")
    print("    -> q = VERDADEIRO")
    print("    -> ~s = VERDADEIRO")
    
    print("\nSe ~s (nao s) e VERDADEIRO, entao 's' deve ser FALSO.")
    print("Este resultado (s = FALSO) e consistente com o que encontramos no Passo 1.")

    # --- Passo 3: Conclusão ---
    p, q, r, s = True, True, False, False
    print("\n\nPasso 3: Conclusao Final")
    print("-" * 70)
    print("Combinando todas as deducoes, os valores verdade sao:")
    print(f"  - p: {str(p).upper()}")
    print(f"  - q: {str(q).upper()}")
    print(f"  - r: {str(r).upper()}")
    print(f"  - s: {str(s).upper()}")

    # --- Passo 4: Verificação Programática ---
    print("\n\nPasso 4: Verificacao Programatica da Solucao")
    print("-" * 70)
    
    # Premissa 1: p -> (r v s) deve ser Falsa
    # Em Python: (not p) or (r or s)
    premissa1_val = (not p) or (r or s)
    print(f"Verificando Premissa 1: (p -> (r v s))")
    print(f"Resultado: {premissa1_val} | Esperado: False -> {'Correto' if premissa1_val is False else 'Incorreto'}")

    # Premissa 2: (q ^ ~s) <-> p deve ser Verdadeira
    # Em Python: ((q and not s) == p)
    premissa2_val = ((q and not s) == p)
    print(f"Verificando Premissa 2: ((q ^ ~s) <-> p)")
    print(f"Resultado: {premissa2_val} | Esperado: True -> {'Correto' if premissa2_val is True else 'Incorreto'}")
    print("=" * 70)

def generate_pdf_report(code_content, output_content):
    """Gera um PDF com o código e o output."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Courier", 'B', 16)
    pdf.cell(0, 10, 'Questao 7', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.ln(10)
    pdf.set_font("Courier", 'B', 12)
    pdf.cell(0, 10, 'Codigo Fonte:', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Courier", '', 8)
    pdf.multi_cell(0, 5, code_content)
    pdf.add_page()
    pdf.set_font("Courier", 'B', 12)
    pdf.cell(0, 10, 'Output da Execucao (Deducao Logica):', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Courier", '', 10)
    output_content_safe = output_content.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 5, output_content_safe)
    pdf_file_name = "resultado_questao_7.pdf"
    pdf.output(pdf_file_name)
    return pdf_file_name

# --- Bloco Principal de Execução ---
if __name__ == "__main__":
    output_buffer = io.StringIO()
    with redirect_stdout(output_buffer):
        solve_and_capture_output()
    output_content = output_buffer.getvalue()

    print("--- [INICIO] Resultado da Execucao no Terminal ---")
    print(output_content)
    print("--- [FIM] Resultado da Execucao no Terminal ---")

    try:
        with open(__file__, 'r', encoding='utf-8') as f:
            code_content = f.read()
    except Exception as e:
        code_content = f"Nao foi possivel ler o arquivo do codigo: {e}"

    try:
        pdf_file = generate_pdf_report(code_content, output_content)
        print(f"\nPDF '{pdf_file}' gerado com sucesso no diretorio atual!")
    except Exception as e:
        print(f"\nOcorreu um erro ao gerar o PDF: {e}")