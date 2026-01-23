#!/usr/bin/env bash
set -euo pipefail

# Run all notebooks in the `notebooks/` directory (excluding checkpoints)
# Outputs are written to `notebooks/executed/`.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NOTEBOOKS_DIR="$ROOT/notebooks"
EXECUTED_DIR="$NOTEBOOKS_DIR/executed"

# Ensure Python can import local `scripts/` package when notebooks run
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

mkdir -p "$EXECUTED_DIR"

echo "Running notebooks from: $NOTEBOOKS_DIR"
echo "Writing executed notebooks to: $EXECUTED_DIR"

# Find notebooks (ignore rough drafts and .ipynb_checkpoints)
find "$NOTEBOOKS_DIR" -maxdepth 2 -type f -name "*.ipynb" \
  ! -path "*/.ipynb_checkpoints/*" \
  -print0 | while IFS= read -r -d '' nb; do
    nb_dir="$(dirname "$nb")"
    nb_base="$(basename "$nb")"
    out="$EXECUTED_DIR/$nb_base"
    echo "Executing: $nb -> $out (running from: $nb_dir)"
    # Run from the notebook directory so relative paths inside notebooks (e.g. ../data/...) resolve correctly
    if command -v papermill >/dev/null 2>&1; then
        (cd "$nb_dir" && papermill "$nb_base" "$out")
    else
        echo "papermill not found; using nbconvert to execute $nb"
        (cd "$nb_dir" && python -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 --output "$out" "$nb_base")
    fi
done

echo "All notebooks executed."
