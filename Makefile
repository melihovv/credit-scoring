#!/bin/bash

MAIN_FILE=explanatory_note

biblio:
	pdflatex $(MAIN_FILE).tex
	bibtex $(MAIN_FILE).aux
	pdflatex $(MAIN_FILE).tex
	pdflatex $(MAIN_FILE).tex

clear:
	rm -f *.pdf *.aux *.log *.gz *.out *.toc *.dvi *.bbl *.blg
