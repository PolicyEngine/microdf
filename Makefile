.PHONY: format lint test install

format:
	linecheck . --fix
	isort --profile black microdf/
	docformatter --wrap-summaries 79 --wrap-descriptions 79 --in-place --recursive microdf/
	black . -l 79

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
	build-changelog changelog.yaml --output changelog.yaml --update-last-date --start-from 0.4.5 --append-file changelog_entry.yaml
	build-changelog changelog.yaml --org PolicyEngine --repo microcalibrate --output CHANGELOG.md --template .github/changelog_template.md
	bump-version changelog.yaml pyproject.toml
	rm changelog_entry.yaml || true
	touch changelog_entry.yaml
