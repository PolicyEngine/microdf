.PHONY: format lint test install

format:
	linecheck . --fix
	isort --profile black microdf/
	black . -l 79

lint:
	linecheck .
	isort --check-only --profile black microdf/
	black . -l 79 --check
	flake8

test:
	pytest -q

install:
	pip install -e .
