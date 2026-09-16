# Sony proposal: LaTeX source

`proposal.tex` is the editable LaTeX version of `proposal_draft.md`. It incorporates the coordinated review revisions and the final creator-facing demonstration: generate a two-person scene, then change the object holder while preserving other requirements. Controlled variation starts with component-specific forward noising of the model's own images and uses the conditional denoiser trained in Section 2.2. Known prompt states remain clean; language-state denoising supplies auxiliary dependency-learning tasks. Representations remain fixed, and no new preliminary experiments or separate editor-training program are required. Keep `references.bib`, `sonyabbrvnat.bst`, and the two active figure PDFs in `figures/` alongside it.

## Compile

Use pdfLaTeX and BibTeX with a standard TeX Live installation. Citations use natbib's author–year format. Mathematical variables are explicitly marked in the Markdown source and exported consistently to LaTeX. The included `sonyabbrvnat.bst` is a lightly modified `abbrvnat` style that links reference titles instead of printing a separate URL.

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error proposal.tex
```

Alternatively, run `pdflatex proposal.tex`, then `bibtex proposal`, then `pdflatex proposal.tex` twice. For Overleaf, upload this directory's files and select pdfLaTeX and `proposal.tex` as the main document.

The checked output is **11 letter-size pages: ten pages of narrative, figures, and references, followed by one budget page**. Body text, tables, and captions use 11-point type; figures use labels of at least 10 points at their seven-inch width, except mathematical subscripts. The PDF is approximately 0.26 MB, with no LaTeX warnings. The four-row hypothesis table stays together. Research prose, equations, references, and benchmark context match the Markdown draft; scholarly citations use BibTeX. Editorial notes are kept outside the proposal.

## Budget and remaining details

The original institution-routed workbook, `budget/Budget_Sony_Cheng.xlsx` at the repository root, requests **USD 144,300**: USD 95,113 direct costs plus USD 49,187 indirect costs. PI effort is 1.5 months. This file is an exact copy of the original upload, preserving every formula, rate, date, and formatting detail. It supersedes the proposed budget revision, which would require another institutional approval. The research plan retains its estimated 5,000 H100-80GB GPU-hours without assigning those estimates to a funding line or quoting cloud prices.

PI email and phone are included with the user's explicit authorization for the private GitHub repository. The original workbook dates and institutional calculations are retained without alteration, as requested. The PI CV remains a separate submission item.

## Maintaining the source

You can edit `proposal.tex` directly. If editing the Markdown instead, run `python scripts/export_latex.py` from the repository root; this overwrites `latex/proposal.tex`, so preserve any direct LaTeX edits first. The conversion requires Pandoc, but compiling the distributed LaTeX does not.

`scripts/make_latex_figures.py` recreates the compact print figures from the existing data and image crops. Their evidence sources and qualifications are documented in the repository's `figures/README.md`; no new experiment was run to create these figures.
