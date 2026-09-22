# Repo-root .latexmkrc
# latexmk reads .latexmkrc from the CURRENT WORKING DIRECTORY (plus ~/.latexmkrc),
# never from the directory of the .tex file. This file makes `latexmk` work from
# the repo root; paper/.latexmkrc handles the case of building inside paper/.

$do_cd = 1;                    # cd into the .tex file's directory before building
do './paper/.latexmkrc';       # single source of truth for the actual settings
@default_files = ('paper/main.tex');
