# Sony proposal: LaTeX source

`proposal.tex` is the editable LaTeX version of `proposal_draft.md`. It incorporates the coordinated review revisions: representations and component groups are fixed before denoising training; schedules and transformer computations are learned within that representation; conditional diagnostics, causal interventions, matched comparisons, and a bounded execution plan connect the method to its claims. Keep `references.bib`, `sonyabbrvnat.bst`, and the two active figure PDFs in `figures/` alongside it.

## Compile

Use pdfLaTeX and BibTeX with a standard TeX Live installation. Citations use natbib's author–year format. Mathematical variables are explicitly marked in the Markdown source and exported consistently to LaTeX. The included `sonyabbrvnat.bst` is a lightly modified `abbrvnat` style that links reference titles instead of printing a separate URL.

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error proposal.tex
```

Alternatively, run `pdflatex proposal.tex`, then `bibtex proposal`, then `pdflatex proposal.tex` twice. For Overleaf, upload this directory's files and select pdfLaTeX and `proposal.tex` as the main document.

The checked output is **10 letter-size pages: nine pages of narrative, figures, and references, followed by one budget page**. Body text, tables, and captions use 11-point type; figures use labels of at least 10 points at their seven-inch width, except mathematical subscripts. The PDF is approximately 0.26 MB, with no LaTeX warnings. The four-row hypothesis table stays together. Research prose, equations, references, and benchmark context match the Markdown draft; scholarly citations use BibTeX. Editorial notes are kept outside the proposal.

## Budget and remaining details

The revised workbook, `budget/Budget_Sony_Cheng_Revised.xlsx` at the repository root, requests **USD 149,999**: USD 98,642 direct costs plus USD 51,357 indirect costs. PI effort is 0.5 months; cloud computing and storage receive USD 23,999. Institutional formulas, rates, other charges, and whole-dollar rounding are preserved. The original uploaded workbook was not modified.

PI email and phone are included with the user's explicit authorization for the private GitHub repository. Confirm the budget calendar dates with the institutional administrator: the workbook lists July 1, 2026–June 30, 2027, while its student rate calculation blends eight months of 2026–27 rates and four months of 2027–28 rates. The proposal uses the revised totals and 12-month duration without inventing replacement dates. The PI CV remains a separate submission item.

## Maintaining the source

You can edit `proposal.tex` directly. If editing the Markdown instead, run `python scripts/export_latex.py` from the repository root; this overwrites `latex/proposal.tex`, so preserve any direct LaTeX edits first. The conversion requires Pandoc, but compiling the distributed LaTeX does not.

`scripts/make_latex_figures.py` recreates the compact print figures from the existing data and image crops. Their evidence sources and qualifications are documented in the repository's `figures/README.md`; no new experiment was run to create these figures.
