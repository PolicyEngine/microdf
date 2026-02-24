.PHONY: format lint test install

format:
	linecheck . --fix
	isort --profile black microdf/
	docformatter --wrap-summaries 79 --wrap-descriptions 79 --in-place --recursive microdf/
	black . -l 79
	@echo "Checking for lines that are too long (E501)..."
	@if flake8 . --select=E501 --max-line-length=79 2>/dev/null | grep -q .; then \
		echo "WARNING: The following lines are too long and must be manually fixed:"; \
		flake8 . --select=E501 --max-line-length=79; \
		echo "Please manually break these long strings/comments to be under 79 characters."; \
		exit 1; \
	fi

lint:
	linecheck .
	isort --check-only --profile black microdf/
	docformatter --wrap-summaries 79 --wrap-descriptions 79 --check --recursive microdf/
	black . -l 79 --check
	flake8

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