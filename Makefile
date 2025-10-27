.PHONY: help setup clean test lint format check verify-dataset quicktest

help:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║  RAG Copy-Out Reproduction - Makefile Commands                ║"
	@echo "║  Reproducing: Follow My Instruction and Spill the Beans       ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "🚀 QUICK START:"
	@echo "  make setup          One-shot installation (detects CUDA/CPU)"
	@echo "  make verify-dataset Test that BM25 indexing works"
	@echo "  make quicktest      Run small-scale test experiment"
	@echo ""
	@echo "🧹 MAINTENANCE:"
	@echo "  make clean          Remove generated files and caches"
	@echo "                      (deletes: datastore/, eval_data/, out/, __pycache__/)"
	@echo ""
	@echo "✅ CODE QUALITY:"
	@echo "  make test           Python syntax checks (py_compile, bash -n)"
	@echo "  make lint           Flake8 code quality checks"
	@echo "  make format         Auto-format with Black (line-length=120)"
	@echo "  make check          Run all checks (test + lint)"
	@echo ""
	@echo "📖 DOCUMENTATION:"
	@echo "  README.md           Quick start and troubleshooting"
	@echo "  REPRO_PLAN.md       6-stage reproduction workflow"
	@echo "  CONTRIBUTING.md     Code style, testing, PR guidelines"
	@echo "  REPO_MAP.md         Architecture and dataflow diagrams"
	@echo ""
	@echo "🔬 EXAMPLE WORKFLOWS:"
	@echo "  # Full reproduction pipeline"
	@echo "  make setup && make verify-dataset"
	@echo "  python scripts/generate_prompts.py --num_samples 100"
	@echo "  python main.py --task io --api hf --hf_ckpt meta-llama/Llama-2-7b-chat-hf ..."
	@echo "  python main.py --task eval --eval_input_dir eval_data/outputs"
	@echo ""
	@echo "  # Development workflow"
	@echo "  make format && make check  # Before committing"
	@echo ""

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

verify-dataset:
	@echo "Verifying BM25 datastore setup..."
	@if [ ! -d "raw_data/private/wiki_newest" ]; then \
		echo "❌ Error: raw_data/private/wiki_newest not found"; \
		echo "Run: python scripts/fetch_wikipedia.py first"; \
		exit 1; \
	fi
	@python scripts/test_datastore.py --raw_data_dir raw_data/private/wiki_newest
	@echo "✓ Dataset verification complete"

quicktest:
	@echo "Running quick test experiment..."
	@echo "This will:"
	@echo "  1. Generate 10 test prompts"
	@echo "  2. Test with tiny model (gpt2)"
	@echo "  3. Verify evaluation pipeline"
	@echo ""
	@mkdir -p prompts eval_data/outputs eval_data/results
	@python scripts/generate_prompts.py --num_samples 10 --output prompts/test_prompts.json
	@echo "Generated test prompts ✓"
	@python main.py --task io \
		--api hf \
		--hf_ckpt gpt2 \
		--is_chat_model false \
		--raw_data_dir raw_data/private/wiki_newest \
		--io_input_path prompts/test_prompts.json \
		--io_output_root eval_data/outputs/test \
		--datastore_root datastore \
		--output_dir out/test \
		--max_new_tokens 50
	@echo "Generated outputs ✓"
	@python main.py --task eval \
		--eval_input_dir eval_data/outputs/test \
		--eval_output_dir eval_data/results/test
	@echo "✓ Quick test complete! Check eval_data/results/test/ for results"
