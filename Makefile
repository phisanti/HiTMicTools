.PHONY: help docs docs-clean docs-serve test test-verbose test-coverage clean clean-all install install-dev lint

# Default Python interpreter
PYTHON := /Users/santiago/miniconda3/envs/img_analysis/bin/python

# Documentation settings
SPHINXOPTS    ?=
SPHINXBUILD   := /Users/santiago/miniconda3/envs/img_analysis/bin/sphinx-build
DOCSOURCEDIR  = docs/source
DOCBUILDDIR   = docs/build

# Colors for terminal output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

help:
	@echo "$(BLUE)HiTMicTools Makefile Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Documentation:$(NC)"
	@echo "  make docs          - Build HTML documentation"
	@echo "  make docs-clean    - Clean documentation build files"
	@echo "  make docs-serve    - Build and serve docs locally (http://localhost:8000)"
	@echo "  make docs-linkcheck - Check for broken links in documentation"
	@echo ""
	@echo "$(GREEN)Testing:$(NC)"
	@echo "  make test          - Run all tests"
	@echo "  make test-verbose  - Run tests with verbose output"
	@echo "  make test-coverage - Run tests with coverage report"
	@echo "  make test-model    - Run model component tests only"
	@echo "  make test-workflow - Run workflow tests only"
	@echo ""
	@echo "$(GREEN)Development:$(NC)"
	@echo "  make install       - Install package in editable mode"
	@echo "  make install-dev   - Install package with dev dependencies"
	@echo "  make lint          - Run code linting (if ruff available)"
	@echo "  make clean         - Clean Python cache files"
	@echo "  make clean-all     - Clean everything (cache, docs, builds)"
	@echo ""

# Documentation commands
docs:
	@echo "$(BLUE)Building documentation...$(NC)"
	@$(SPHINXBUILD) -M html "$(DOCSOURCEDIR)" "$(DOCBUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "$(GREEN)Documentation built successfully!$(NC)"
	@echo "Open: $(DOCBUILDDIR)/html/index.html"

docs-clean:
	@echo "$(YELLOW)Cleaning documentation build files...$(NC)"
	@rm -rf $(DOCBUILDDIR)/*
	@echo "$(GREEN)Documentation cleaned$(NC)"

docs-serve: docs
	@echo "$(BLUE)Starting local documentation server...$(NC)"
	@echo "$(GREEN)Documentation available at: http://localhost:8000$(NC)"
	@echo "Press Ctrl+C to stop"
	@cd $(DOCBUILDDIR)/html && $(PYTHON) -m http.server 8000

docs-linkcheck:
	@echo "$(BLUE)Checking documentation links...$(NC)"
	@$(SPHINXBUILD) -M linkcheck "$(DOCSOURCEDIR)" "$(DOCBUILDDIR)" $(SPHINXOPTS) $(O)

# Testing commands
test:
	@echo "$(BLUE)Running all tests...$(NC)"
	@$(PYTHON) -m unittest discover tests -v

test-verbose:
	@echo "$(BLUE)Running tests with verbose output...$(NC)"
	@$(PYTHON) -m unittest discover tests -v

test-coverage:
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	@$(PYTHON) -m coverage run -m unittest discover tests
	@$(PYTHON) -m coverage report
	@$(PYTHON) -m coverage html
	@echo "$(GREEN)Coverage report generated: htmlcov/index.html$(NC)"

test-model:
	@echo "$(BLUE)Running model component tests...$(NC)"
	@$(PYTHON) -m unittest discover tests/model_components -v

test-workflow:
	@echo "$(BLUE)Running workflow tests...$(NC)"
	@$(PYTHON) -m unittest discover tests/tests_workflows -v

# Development commands
install:
	@echo "$(BLUE)Installing HiTMicTools in editable mode...$(NC)"
	@$(PYTHON) -m pip install -e .
	@echo "$(GREEN)Installation complete$(NC)"

install-dev:
	@echo "$(BLUE)Installing HiTMicTools with development dependencies...$(NC)"
	@$(PYTHON) -m pip install -e .
	@$(PYTHON) -m pip install pytest coverage ruff sphinx sphinx-rtd-theme myst-parser
	@echo "$(GREEN)Development installation complete$(NC)"

lint:
	@echo "$(BLUE)Running linter...$(NC)"
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check src tests; \
	else \
		echo "$(YELLOW)Ruff not installed. Install with: pip install ruff$(NC)"; \
	fi

# Cleanup commands
clean:
	@echo "$(YELLOW)Cleaning Python cache files...$(NC)"
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)Python cache cleaned$(NC)"

clean-all: clean docs-clean
	@echo "$(YELLOW)Cleaning all build artifacts...$(NC)"
	@rm -rf dist/
	@rm -rf build/
	@rm -rf htmlcov/
	@rm -rf .coverage
	@echo "$(GREEN)All artifacts cleaned$(NC)"
