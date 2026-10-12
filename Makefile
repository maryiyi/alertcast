.PHONY: test train run all

test:
	.venv/bin/pytest -v tests/

train:
	.venv/bin/python train_and_eval.py

run: train

all: test train
