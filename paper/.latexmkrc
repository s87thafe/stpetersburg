# paper/.latexmkrc

$pdf_mode = 1;                 # build PDF via pdflatex
$bibtex_use = 2;               # run BibTeX when needed; delete .bbl on clean
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
$out_dir = 'build';            # aux files + PDF go to build/
@default_files = ('main.tex');