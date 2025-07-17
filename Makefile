.PHONY: format lint test install

format:
	linecheck . --fix
	black . -l 79

lint:
	flake8

test:
	pytest -q

install:
	pip install -e .
