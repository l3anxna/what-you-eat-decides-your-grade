# what-you-eat-decides-your-grade

Predicts GPA from a food & lifestyle survey. See `EDA_notebook.ipynb` for the analysis and `INTERPRETATION.md` for the findings.

## Setup

```
git clone https://github.com/l3anxna/what-you-eat-decides-your-grade.git
cd what-you-eat-decides-your-grade
uv sync --group ml
```

## Run

```
uv run python scr/pipeline.py   # trains & saves all 5 models to models/
uv run python main.py           # interactive CLI: pick a model, predict on sample data
```

`uv run pytest` runs the tests. `uv run --group eda --group ml python scr/analytic.py` regenerates the plots in `outputs/`.

## Docker

```
docker build -t what-you-eat-decides-your-grade .
docker run --rm -it what-you-eat-decides-your-grade
```

The build trains and bakes all 5 models into the image (runs `scr/pipeline.py`). `-it` is required for the interactive model-selection prompt; without it, the container just exits cleanly.

## Members

| **Github**       | **Classroom**                |
|------------------|------------------------------|
| \l3anxna         | Patthadon Aroonpairodjanakul |
| QuantumTacoNinja | Worapat Hongsetong           |
