# Notebooks

Notebooks are for **exploration and communication**, not for production logic.

Rules:
1. Name them `NN_topic.ipynb` (e.g. `01_eda.ipynb`, `02_baseline.ipynb`).
2. Import project code instead of copying it:
   `from <package>.features.build import build_features`.
3. When a cell becomes useful beyond the notebook, move it to `src/` and test it.
4. Outputs are stripped automatically on commit (`nbstripout` pre-commit hook),
   so diffs stay readable and no data leaks into git.
5. Select the project kernel: `.venv` created by `uv sync`.
