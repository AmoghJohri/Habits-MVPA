SHELL := /bin/bash

.PHONY: env install run-notebooks figures clean

env:
	conda env create -f environment.yml || echo "Environment may already exist"

install:
	pip install -r requirements.txt

run-notebooks:
	@echo "Executing notebooks (this may take a while)..."
	mkdir -p notebooks/executed
	for nb in notebooks/*.ipynb; do \
	  echo "Running $$nb..."; \
	  jupyter nbconvert --to notebook --execute "$$nb" --output "notebooks/executed/$$(basename $$nb)" || exit 1; \
	done

# Run the project-level script that executes notebooks and generates figures.
.PHONY: figures
figures:
	@echo "Running scripts/make_figures.sh to execute all notebooks and produce figures..."
	./scripts/make_figures.sh
	@echo "Done — check notebooks/executed/ and figures/."

clean:
	rm -rf notebooks/executed
