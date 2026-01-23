# Reproducibility Guide

This document explains how to reproduce the analyses and figures in this repository.

1. Install the environment using `environment.yml` or install packages via `requirements.txt`.

2. Make sure raw data is placed in the `data/` folder. If data cannot be shared, provide instructions here for obtaining or requesting access.

3. Run the notebooks with:

```bash
make figures
# or
./scripts/run_notebooks.sh
```

4. Executed notebooks are saved to `notebooks/executed/`. Figures produced by notebooks should be saved to `figures/` by each notebook or collected via `scripts/make_figures.py`.

Notes
-----

- For long-running computations, consider caching intermediate results to `data/interim/` and adding a `data/README.md` describing provenance.
- To automate checks, add a GitHub Actions workflow that runs `make figures`.
