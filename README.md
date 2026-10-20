# Biomanifolds
Experiments on feature manifolds in ESM-2 (and other models, later)

## Setup
- install with `uv sync --dev --extra cuda`
    - if you have an AMD GPU, do `uv sync --dev --extra rocm`
    - the code will run fine independently of what GPU you have. It won't run on CPU, though.

- if you don't have `uv` try this, but I haven't tested it
```
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128 
pip install -e .
pip install ipykernel jupyter 
```
- for AMD GPU replace torch line with `pip install torch --index-url https://download.pytorch.org/whl/rocm7.2`

## Project structure: 
- `notebooks`: to be run in order, some depend on results from previous notebooks. There's no script to reproduce everything in one click yet.
- `biomanifolds`
    |
    |-- `datasets`: helper functions to download data, preprocess, and reproduce our train/test split
    |
    |-- `labels`: code that defines how to compute each label derived from sequence, structure, and swissprot annotations in separate modules. Each label has its own function, for readability. 

