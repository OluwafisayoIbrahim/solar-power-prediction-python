# Manuscript and editable source

`editorial_reference_superseded.zip` is an unchanged copy of the latest edited
LaTeX bundle supplied for the manuscript. It contains editable LaTeX, the old
PDF, seven source tables, main and supplementary figure PDFs, and Vega-Lite
specifications. The title and six updated reference attributions must be
preserved. This bundle uses superseded 189-row results and is not an updated
scientific output.

The updated LaTeX/PDF follows the completed experiments under
`results`, retaining the supplied editorial improvements
and including the recalculated no-fitting voltage/current product in the abstract.
No collection month/year can be inferred from filenames. Small differences
remain descriptive without a significance analysis. All affected findings,
fold counts, diagrams and charts have been regenerated.

`source/manuscript.pdf` is the nine-page rendered manuscript. The adjacent
`manuscript.tex`, seven table CSVs, main and supplementary figure PDFs, and
Vega-Lite specifications form its editable source. `solar_manuscript_latex.zip`
packages these files. `publication_ready.json` records the content and visual review.

To regenerate, install `vl-convert-python` in addition to the research environment,
run `python scripts/build_manuscript.py`, then compile `source/manuscript.tex`
with a LaTeX distribution supporting IEEEtran and newtx. Review every rendered
page after regeneration. Opening the app does not compile or train anything.
