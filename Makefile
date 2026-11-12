PY ?= python

.PHONY: setup data train test lint format clean

setup:
	$(PY) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/python -m ipykernel install --user --name superstore --display-name "Python (superstore)"

data:
	$(PY) scripts/make_dataset.py

train:
	$(PY) scripts/run_experiments.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src scripts tests notebooks

format:
	$(PY) -m ruff format src scripts tests
	$(PY) -m ruff check --fix src scripts tests

clean:
	rm -rf data/interim/*.parquet data/processed/*.parquet .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
