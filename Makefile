.PHONY: format lint test

format:
	black microdf

lint:
	flake8

test:
	pytest -q
