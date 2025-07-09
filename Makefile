.PHONY: format lint test install

format:
	black microdf

lint:
	flake8

test:
	pytest -q

install:
	pip install -e .
