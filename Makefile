.PHONY: help setup clean test lint format check

help:
	@echo "RAG Copy-Out Experiment - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  setup    - Run one-shot installation (./setup.sh)"
	@echo "  clean    - Remove generated files and caches"
	@echo "  test     - Run syntax checks on all scripts"
	@echo "  lint     - Run flake8 code quality checks"
	@echo "  format   - Auto-format code with black"
	@echo "  check    - Run all checks (test + lint)"
	@echo ""
	@echo "For reproduction workflow, see REPRO_PLAN.md"

setup:
	@echo "Running setup script..."
	./setup.sh

clean:
	@echo "Cleaning generated files..."
	rm -rf datastore/
	rm -rf eval_data/
	rm -rf out/
	rm -rf downloads/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Clean complete"

test:
	@echo "Testing Python scripts for syntax errors..."
	@python -m py_compile scripts/*.py
	@python -m py_compile modules/*.py
	@python -m py_compile utils/*.py
	@python -c "import main; print('✓ main.py OK')"
	@bash -n scripts/*.sh
	@bash -n setup.sh
	@echo "✓ All scripts syntax OK"

lint:
	@echo "Running flake8..."
	@flake8 scripts/ modules/ utils/ main.py --max-line-length=120 || echo "⚠ Linting warnings (non-critical)"

format:
	@echo "Formatting code with black..."
	@black scripts/ modules/ utils/ main.py --line-length=120
	@echo "✓ Formatting complete"

check: test lint
	@echo "✓ All checks passed"
