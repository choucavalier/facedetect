MD_FILES=facedetect

all:
	for f in ${MD_FILES}; do pandoc $$f.md -o $$f.tex; done
	pdflatex -shell-escape src
	makeglossaries src
	bibtex src
	pdflatex -shell-escape src
	pdflatex -shell-escape src

clean:
	for f in ${MD_FILES}; do rm -f $$f.tex; rm -f $$f.aux; done
	rm -f *.toc *.aux *.gls *.glo *.glg *.alg *.acn *xdy *.log *.acr *.out *.blg *.bbl *.sta
	rm -rf _minted-src
