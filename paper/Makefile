# Build the draft.
#
# No TeX is on the default PATH on this cluster. TeX Live 2020 lives at the
# path below; adjust if you build elsewhere.
TEXBIN = /N/soft/rhel8/texlive/2020/bin/x86_64-linux
export PATH := $(TEXBIN):$(PATH)

MAIN = main

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex refs.bib $(wildcard sections/*.tex)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

# Hide \todo markers for circulation
final:
	sed -i.bak 's/^\\draftmodetrue/\\draftmodefalse/' $(MAIN).tex
	latexmk -pdf -interaction=nonstopmode $(MAIN).tex
	mv $(MAIN).tex.bak $(MAIN).tex

clean:
	latexmk -C
	rm -f *.bbl *.blg *.aux *.log *.out

.PHONY: all final clean
