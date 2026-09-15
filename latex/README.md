# Sony proposal: LaTeX source

`proposal.tex` is the editable LaTeX version of the current `proposal_draft.md`, including its updated institutional budget. The four figure PDFs in `figures/` must remain alongside it.

## Compile

Use pdfLaTeX with a standard TeX Live installation. No BibTeX run is needed: references and hyperlinks are included directly in the source.

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error proposal.tex
```

Alternatively, run `pdflatex proposal.tex` twice. For Overleaf, upload this directory's files and select pdfLaTeX and `proposal.tex` as the main document.

The checked output is 11 letter-size pages: ten pages of narrative, figures, and references, followed by one budget page. Body text, tables, and captions use 10-point type; figures use labels of at least 10 points at their specified seven-inch width, except normal mathematical subscripts. The PDF is approximately 0.3 MB. All research prose, equations, references, and evidence qualifications are retained. Internal drafting notes appear after `\end{document}` as comments and are not printed.

## Budget and remaining details

The supplied `Budget_Sony_Cheng.xlsx` yields a request of USD 144,300: USD 95,113 direct costs plus USD 49,187 indirect costs. Its line items, effort, rates, and whole-dollar rounding have been preserved. The original workbook was not modified.

Before submission, complete the PI email and phone placeholders. Confirm the budget calendar dates with the institutional administrator: the workbook lists July 1, 2026–June 30, 2027, while its student rate calculation blends eight months of 2026–27 rates and four months of 2027–28 rates. The proposal uses the supplied totals and 12-month duration without inventing replacement dates. The PI CV remains a separate submission item.

## Maintaining the source

You can edit `proposal.tex` directly. If editing the Markdown instead, run `python scripts/export_latex.py` from the repository root; this overwrites `latex/proposal.tex`, so preserve any direct LaTeX edits first. The conversion requires Pandoc, but compiling the distributed LaTeX does not.

`scripts/make_latex_figures.py` recreates the compact print figures from the existing data and image crops. Their evidence sources and qualifications are documented in the repository's `figures/README.md`; no new experiment was run to create these figures.
