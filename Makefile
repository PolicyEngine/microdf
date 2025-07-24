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
	build-changelog changelog.yaml --output changelog.yaml --update-last-date --start-from 0.4.5 --append-file changelog_entry.yaml
	build-changelog changelog.yaml --org PolicyEngine --repo microcalibrate --output CHANGELOG.md --template .github/changelog_template.md
	bump-version changelog.yaml pyproject.toml
	rm changelog_entry.yaml || true
	touch changelog_entry.yaml
