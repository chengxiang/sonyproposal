#!/usr/bin/env python3
"""Convert the current Markdown proposal to editable, self-contained LaTeX.

Requires Pandoc. The generated source compiles independently with pdfLaTeX;
Pandoc and this script are not needed by the recipient of the source bundle.
Research prose is retained; internal drafting notes become LaTeX comments.
"""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'latex'
OUT.mkdir(exist_ok=True)
source = (ROOT / 'proposal_draft.md').read_text()
definitions = '\n'.join(re.findall(r'^\[[^\]]+\]: .+$', source, re.M))
main, editorial = source.split('## Editorial notes for finalization', 1)
editorial = editorial.split('\n[dino]:', 1)[0]
draft_note = re.search(r'^\*Drafting note.*?\*$', main, re.M).group(0)
main = main.replace(draft_note, '')
title, main = main.split('\n', 1)
title = title.removeprefix('# ')
bib_keys = set(re.findall(r'^@\w+\{([^,]+),', (OUT / 'references.bib').read_text(), re.M))

# The Markdown remains readable on GitHub; scholarly links become natbib citations.
def citations(md):
    return re.sub(r'\[([^\]]+)\]\[([^\]]+)\]',
                  lambda m: m[1] + r' \citep{' + m[2] + '}'
                  if m[2] in bib_keys else m[0], md)

# Retain the full reference database; the bibliography is formatted by BibTeX.
main = re.sub(r'## References\n.*?(?=\*\*Unpublished preliminary materials\.)',
              lambda _: '\\bibliographystyle{sonyabbrvnat}\n\\nocite{*}\n\\bibliography{references}\n\n',
              main, flags=re.S)

in_math = False
prose_lines = []
for line in main.splitlines():
    if line.startswith('```math'):
        in_math = True
    elif line.startswith('```'):
        in_math = False
    elif not in_math:
        line = line.replace('v_k=s_k−ε_k', r'\(v_k=s_k-\varepsilon_k\)')
        line = re.sub(r'(?<![\w\\])(?:P_k|d_k|I_m|O_m|g_m|u_j|f_k)(?![\w])',
                      lambda m: r'\(' + m[0] + r'\)', line)
    prose_lines.append(line)
main = '\n'.join(prose_lines)

def normalize(text):
    replacements = {
        '1 − E_hybrid/E_source': r'\(1-E_{\mathrm{hybrid}}/E_{\mathrm{source}}\)',
        'fθ,i': r'\(f_{\theta,i}\)', 'fθ': r'\(f_\theta\)',
        'θₘ': r'\(\theta_m\)', 'τₖ': r'\(\tau_k\)',
        'sₖ=Pₖ(Z)': r'\(s_k=P_k(Z)\)',
        'sₖ': r'\(s_k\)', 'Pₖ': r'\(P_k\)',
        'a₀': r'\(a_0\)', 'f₀': r'\(f_0\)',
        'θ': r'\(\theta\)', 'τ': r'\(\tau\)', 'Σ': r'\(\Sigma\)',
        'φ': r'\(\phi\)', 'ψ': r'\(\psi\)', 'ε': r'\(\varepsilon\)',
        '−': r'\(-\)', '→': r'\(\to\)',
        '—': '---', '–': '--', '“': '"', '”': '"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def convert(md):
    return subprocess.run([
        'pandoc', '-f', 'markdown+raw_tex+tex_math_single_backslash',
        '-t', 'latex', '--wrap=none', '--shift-heading-level-by=-1',
    ], input=normalize(citations(md)) + '\n\n' + definitions, text=True,
       capture_output=True, check=True).stdout.strip()

def math_block(match):
    return '\n\\[\n' + match.group(1).strip() + '\n\\]\n'

main = re.sub(r'```math\n(.*?)\n```', math_block, main, flags=re.S)

def figure(match):
    path, n, caption = match.group(1), match.group(2), match.group(3)
    # The explicit bold title and all evidence qualifications are retained.
    caption = convert('**' + caption)
    pdf_path = 'figures/' + Path(path).stem + '.pdf'
    placement = '!htbp' if n == '1' else 'H'
    return ('\n\\begin{figure}[' + placement + ']\n\\centering\n'
            + r'\includegraphics[width=7in]{' + pdf_path + '}\n'
            + r'\caption{' + caption + '}\n'
            + r'\label{fig:' + n + '}\n\\end{figure}\n')

main = re.sub(r'!\[[^\]]*\]\(([^)]+)\)\n\n\*\*Figure (\d+)\. ([^\n]+)',
              figure, main)
main = main.replace('## Budget summary', '\\clearpage\n\n## Budget summary')
body = convert(main)

# Set widths by the table's information content, rather than Markdown dashes.
def table_widths(match):
    table = match.group(0)
    if 'Budget category' in table:
        widths = [.80, .20]
    elif 'Concrete output' in table:
        widths = [.14, .47, .39]
    elif 'What it establishes' in table:
        widths = [.24, .43, .33]
    elif 'Role in the initial implementation' in table:
        widths = [.33, .67]
    else:
        return table
    index = iter(widths)
    return re.sub(r'\\real\{[\d.]+\}',
                  lambda _: r'\real{' + f'{next(index):.4f}' + '}', table)

body = re.sub(r'\\begin\{longtable\}.*?\\end\{longtable\}', table_widths, body, flags=re.S)

preamble = r'''% Generated from proposal_draft.md; edit this file directly if preferred.
% Compile: latexmk -pdf -interaction=nonstopmode -halt-on-error proposal.tex
\documentclass[10pt,letterpaper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{mathptmx}
\usepackage{amsmath,amssymb}
\usepackage[letterpaper,margin=0.65in]{geometry}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{float}
\usepackage{array,booktabs,longtable,calc}
\usepackage{caption}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{xcolor}
\usepackage{xurl}
\usepackage[authoryear,round]{natbib}
\usepackage[hidelinks,unicode]{hyperref}
\setlength{\bibsep}{1.5pt}
\renewcommand{\bibfont}{\normalfont\normalsize}
\hypersetup{pdftitle={How Representations and Weights Shape Multimodal Generation},pdfauthor={Xiang Cheng}}
\urlstyle{same}
\setcounter{secnumdepth}{-2}
\setlength{\parindent}{0pt}
\linespread{0.96}
\setlength{\parskip}{2.5pt plus 0.5pt minus 0.3pt}
\setlength{\emergencystretch}{1.8em}
\titleformat{\section}{\large\bfseries}{}{0pt}{}
\titleformat{\subsection}{\normalsize\bfseries}{}{0pt}{}
\titlespacing*{\section}{0pt}{9pt plus 2pt minus 1pt}{4pt}
\titlespacing*{\subsection}{0pt}{7pt plus 1pt minus 1pt}{3pt}
\setlist{leftmargin=*,itemsep=1.5pt,topsep=3pt,parsep=0pt}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\captionsetup{font=normalsize,labelfont=bf,skip=5pt}
\setlength{\textfloatsep}{9pt plus 2pt minus 2pt}
\setlength{\intextsep}{8pt plus 2pt minus 2pt}
\setlength{\floatsep}{8pt plus 2pt minus 2pt}
\renewcommand{\topfraction}{0.95}
\renewcommand{\bottomfraction}{0.9}
\renewcommand{\textfraction}{0.05}
\renewcommand{\floatpagefraction}{0.85}
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{3}
\setcounter{totalnumber}{5}
\setlength{\LTpre}{5pt}
\setlength{\LTpost}{5pt}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.06}
\makeatletter
\def\fps@figure{htbp}
\makeatother
\begin{document}
\begin{center}
{\LARGE\bfseries How Representations and Weights Shape\\[2pt] Multimodal Generation\par}
\end{center}
\vspace{-5pt}
'''

notes = '\n'.join('% ' + l for l in (draft_note + '\n\n' + editorial).splitlines())
(OUT / 'proposal.tex').write_text(preamble + '\n' + body
    + '\n\n\\end{document}\n\n% Internal drafting notes (not typeset):\n' + notes + '\n')
print(OUT / 'proposal.tex')
