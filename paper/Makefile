# Build the paper with tectonic (https://tectonic-typesetting.github.io); it runs bibtex and reruns as needed.
TECTONIC ?= $(HOME)/.local/bin/tectonic

main.pdf: main.tex refs.bib figures/f1_accuracy_vs_tokens.pdf figures/f2_library_vs_alternatives.pdf
	$(TECTONIC) --keep-logs main.tex

figures/%.pdf: ../scripts/paper/figures.py ../results/analysis/partd_f.json ../results/analysis/partb_test_embed-lr.json
	cd .. && .venv-paper/bin/python scripts/paper/figures.py

clean:
	rm -f main.pdf main.log main.aux main.bbl main.blg main.out

.PHONY: clean
