.PHONY: format lint test install

format:
	docformatter --wrap-summaries 79 --wrap-descriptions 79 --in-place --recursive microdf/
	ruff format .

lint:
	docformatter --wrap-summaries 79 --wrap-descriptions 79 --check --recursive microdf/
	ruff format --check .

test:
	pytest -q --cov=microdf --cov-report=xml

install:
	pip install -e ".[dev]"

build:
	pip install build
	python -m build --wheel --sdist

changelog:
	python .github/bump_version.py
	towncrier build --yes --version $$(python -c "import re; print(re.search(r'version = \"(.+?)\"', open('pyproject.toml').read()).group(1))")