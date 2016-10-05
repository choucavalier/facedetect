all:
	pdflatex figure.tex -o figure.pdf
	pdf2svg figure.pdf figure.svg
	make clean

clean:
	rm figure.pdf
	rm figure.aux
	rm figure.log
