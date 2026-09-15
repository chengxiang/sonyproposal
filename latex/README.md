# Sony proposal: LaTeX source

`proposal.tex` is the editable LaTeX version of the current `proposal_draft.md`, organized as three connected research sections (2.1–2.3), with a dedicated state-of-the-art differentiation section and the institutional budget unchanged. Keep `references.bib`, `sonyabbrvnat.bst`, and the four figure PDFs in `figures/` alongside it.

## Compile

Use pdfLaTeX and BibTeX with a standard TeX Live installation. Citations use natbib's author–year format. The included `sonyabbrvnat.bst` is a lightly modified `abbrvnat` style that links reference titles instead of printing a separate URL.

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error proposal.tex
```

Alternatively, run `pdflatex proposal.tex`, then `bibtex proposal`, then `pdflatex proposal.tex` twice. For Overleaf, upload this directory's files and select pdfLaTeX and `proposal.tex` as the main document.

The checked output is 10 letter-size pages: nine pages of narrative, figures, and references, followed by one budget page. Body text, tables, and captions use 10-point type; figures use labels of at least 10 points at their specified seven-inch width, except normal mathematical subscripts. The PDF is approximately 0.3 MB. Research prose, equations, references, and benchmark context match the current Markdown draft; scholarly citations are formatted through BibTeX. Internal drafting notes appear after `\end{document}` as comments and are not printed.

## Budget and remaining details

The supplied `Budget_Sony_Cheng.xlsx` yields a request of USD 144,300: USD 95,113 direct costs plus USD 49,187 indirect costs. Its line items, effort, rates, and whole-dollar rounding have been preserved. The original workbook was not modified.

Before submission, complete the PI email and phone placeholders. Confirm the budget calendar dates with the institutional administrator: the workbook lists July 1, 2026–June 30, 2027, while its student rate calculation blends eight months of 2026–27 rates and four months of 2027–28 rates. The proposal uses the supplied totals and 12-month duration without inventing replacement dates. The PI CV remains a separate submission item.

## Maintaining the source

You can edit `proposal.tex` directly. If editing the Markdown instead, run `python scripts/export_latex.py` from the repository root; this overwrites `latex/proposal.tex`, so preserve any direct LaTeX edits first. The conversion requires Pandoc, but compiling the distributed LaTeX does not.

`scripts/make_latex_figures.py` recreates the compact print figures from the existing data and image crops. Their evidence sources and qualifications are documented in the repository's `figures/README.md`; no new experiment was run to create these figures.
