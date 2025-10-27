# Contributing to RAG Copy-Out Reproduction

Thank you for your interest in contributing to this reproduction of "Follow My Instruction and Spill the Beans" (ICLR 2025)! This document provides guidelines for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Requirements](#testing-requirements)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)
- [Project Structure](#project-structure)

## 🤝 Code of Conduct

This project is focused on **legitimate security research and reproducible science**. We expect contributors to:

- Focus on **defensive security** and understanding vulnerabilities
- Use this research responsibly and ethically
- Respect privacy and security boundaries
- Cite sources properly and maintain scientific integrity
- Be respectful in all interactions

## 🚀 Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/yourusername/2881-mini-project.git
cd 2881-mini-project
```

### 2. Set Up Environment

```bash
# One-command setup
./setup.sh

# Or manually with conda
conda env create -f environment.yml
conda activate rag-privacy
```

### 3. Verify Installation

```bash
# Run quick datastore test
python scripts/test_datastore.py --raw_data_dir raw_data/private/wiki_newest

# Run linting checks
make lint

# Run all checks
make check
```

## 🔄 Development Workflow

### Branch Naming

Use descriptive branch names following these patterns:

- `feature/description` - New features (e.g., `feature/add-faiss-retrieval`)
- `fix/issue-number` - Bug fixes (e.g., `fix/repetition-penalty-bug`)
- `docs/description` - Documentation updates (e.g., `docs/update-readme`)
- `refactor/component` - Code refactoring (e.g., `refactor/lm-module`)
- `experiment/description` - Experimental features (e.g., `experiment/new-attack-prompt`)

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/) specification:

```bash
# Format
<type>(<scope>): <description>

# Examples
feat(retrieval): Add FAISS-based kNN retrieval option
fix(generation): Add missing repetition_penalty parameter
docs(readme): Update troubleshooting section
refactor(utils): Simplify metrics computation
test(datastore): Add BM25 integration tests
chore(deps): Update transformers to 4.36.2
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style/formatting (no functional changes)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks (dependencies, configs)
- `perf`: Performance improvements

## 🎨 Code Style Guidelines

### Python Code Style

**Follow PEP 8** with these specific guidelines:

```python
# ✅ GOOD: Clear variable names, type hints, docstrings
def compute_rouge_l(
    predictions: List[str],
    references: List[str],
    use_stemmer: bool = True
) -> Dict[str, float]:
    """
    Compute ROUGE-L scores between predictions and references.

    Args:
        predictions: List of generated text strings
        references: List of ground truth text strings
        use_stemmer: Whether to use Porter stemmer (default: True)

    Returns:
        Dictionary with 'precision', 'recall', 'fmeasure' keys
    """
    # Implementation here
    pass

# ❌ BAD: No type hints, unclear names, no docstring
def comp_rl(p, r, s=True):
    # what does this do?
    pass
```

**Key rules:**
- **Line length**: 88 characters (Black formatter default)
- **Indentation**: 4 spaces (no tabs)
- **Imports**: Group into stdlib, third-party, local (separated by blank lines)
- **Type hints**: Required for all function signatures
- **Docstrings**: Required for all public functions/classes (use Google style)
- **Naming**:
  - Functions/variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`
  - Private members: `_leading_underscore`

### Formatting Tools

Use these tools before committing:

```bash
# Auto-format with Black
python -m black .

# Sort imports with isort
python -m isort .

# Type checking with mypy
python -m mypy modules/ utils/

# All checks in one command
make check
```

### Code Organization

- **Keep functions small**: Max 60 lines, single responsibility
- **Avoid deep nesting**: Max 3-4 levels of indentation
- **No magic numbers**: Use named constants
- **Error handling**: Always handle exceptions explicitly
- **Comments**: Explain "why", not "what"

```python
# ✅ GOOD: Named constant, clear intent
MAX_RETRIEVAL_DOCS = 5
docs = retriever.get_top_k(query, k=MAX_RETRIEVAL_DOCS)

# ❌ BAD: Magic number
docs = retriever.get_top_k(query, k=5)  # why 5?
```

## 🧪 Testing Requirements

### Test Structure

```
tests/
├── unit/              # Fast, isolated tests
│   ├── test_metrics.py
│   ├── test_chunker.py
│   └── test_prompt_builder.py
├── integration/       # Component interaction tests
│   ├── test_bm25_retrieval.py
│   └── test_lm_generation.py
└── fixtures/          # Test data
    └── sample_docs.json
```

### Writing Tests

```python
import pytest
from utils.metrics import compute_rouge_l

def test_rouge_l_with_identical_strings():
    """ROUGE-L should be 1.0 for identical strings."""
    predictions = ["The quick brown fox"]
    references = ["The quick brown fox"]

    scores = compute_rouge_l(predictions, references)

    assert scores['fmeasure'] == pytest.approx(1.0, abs=1e-6)

def test_rouge_l_with_empty_prediction():
    """ROUGE-L should be 0.0 for empty prediction."""
    predictions = [""]
    references = ["Some reference text"]

    scores = compute_rouge_l(predictions, references)

    assert scores['fmeasure'] == 0.0
```

**Test naming convention:**
- Format: `test_<function>_<scenario>`
- Be descriptive: `test_should_return_zero_when_input_empty`
- One assertion per test (when possible)

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=modules --cov=utils

# Run specific test file
pytest tests/unit/test_metrics.py

# Run with verbose output
pytest -v

# Using Makefile
make test
```

### Coverage Guidelines

- **Aim for meaningful coverage**, not 100%
- **Must test**: Public APIs, edge cases, error paths
- **Can skip**: Private helpers, trivial getters/setters
- **Required**: All new features must include tests

## 📝 Submitting Changes

### Pull Request Process

1. **Create feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes and commit**:
   ```bash
   git add .
   git commit -m "feat(scope): clear description"
   ```

3. **Run all checks locally**:
   ```bash
   make check  # Runs lint, format check, type check, tests
   ```

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open Pull Request** on GitHub with this template:

   ```markdown
   ## Summary
   Brief description of what this PR does and why.

   ## Changes
   - Added X functionality to Y module
   - Fixed Z bug in W component
   - Updated documentation for V

   ## Testing
   - [ ] Ran `make check` locally (all passed)
   - [ ] Added unit tests for new functionality
   - [ ] Tested on sample dataset
   - [ ] Verified no regression on existing tests

   ## Related Issues
   Closes #123

   ## Screenshots (if applicable)
   ```

### Pull Request Checklist

Before submitting, ensure:

- [ ] Code follows style guidelines (Black, isort)
- [ ] All tests pass (`pytest`)
- [ ] Type hints added for new functions (`mypy`)
- [ ] Docstrings added for public APIs
- [ ] Documentation updated (README, REPRO_PLAN if needed)
- [ ] No hardcoded secrets or API keys
- [ ] Commit messages follow Conventional Commits
- [ ] Branch is up to date with `main`

### Review Process

- Maintainers will review within 3-5 days
- Address feedback with new commits (don't force-push during review)
- Once approved, maintainers will merge

## 🐛 Reporting Issues

### Bug Reports

Use this template when filing bug reports:

```markdown
**Describe the bug**
Clear description of what went wrong.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. See error '...'

**Expected behavior**
What you expected to happen.

**Environment**
- OS: [e.g. Ubuntu 22.04]
- Python: [e.g. 3.10.13]
- PyTorch: [e.g. 2.1.2+cu118]
- CUDA: [e.g. 11.8] or CPU-only

**Error output**
```
Paste full error traceback here
```

**Additional context**
Any other relevant information.
```

### Feature Requests

Use this template for feature requests:

```markdown
**Feature description**
Clear description of the proposed feature.

**Motivation**
Why is this feature needed? What problem does it solve?

**Proposed solution**
How should this feature work?

**Alternatives considered**
What other approaches did you consider?

**Additional context**
Related papers, links, examples.
```

## 📂 Project Structure

Understanding the codebase organization:

```
.
├── main.py              # Main entry point
├── modules/             # Core modules
│   ├── LM.py           # Language model wrapper
│   ├── Retriever.py    # BM25 retrieval
│   ├── Prompter.py     # Prompt building (RIC-LM)
│   └── TogetherAI_API.py  # Together.ai API client (optional)
├── utils/               # Utility functions
│   ├── chunker.py      # Text chunking
│   ├── metrics.py      # Evaluation metrics
│   └── evaluator.py    # Evaluation orchestration
├── scripts/             # Helper scripts
│   ├── fetch_wikipedia.py        # Data collection
│   ├── generate_prompts.py       # Prompt generation
│   ├── test_datastore.py         # Setup verification
│   ├── run_all_models.sh         # Batch runner
│   └── generate_results_table.py # Results formatting
├── tests/               # Test suite
├── docs/                # Additional documentation
├── raw_data/            # Wikipedia articles
├── datastore/           # BM25 indices
├── prompts/             # Attack prompts
├── eval_data/           # Experiment outputs
└── out/                 # Model outputs
```

### Key Files to Know

- **main.py**: CLI entry point for all tasks (io, eval)
- **modules/LM.py**: Handles HuggingFace and Together API model inference
- **modules/Retriever.py**: BM25 indexing and retrieval with Pyserini
- **modules/Prompter.py**: Implements RIC-LM prompt construction
- **utils/metrics.py**: ROUGE-L, BLEU, F1, BERTScore computation
- **utils/chunker.py**: 256-token chunking with 128-token stride
- **scripts/fetch_wikipedia.py**: Wikipedia data collection from Cirrus dumps

### Adding New Features

**Adding a new retrieval method (e.g., FAISS):**

1. Create `modules/FAISSRetriever.py`
2. Follow interface from `modules/Retriever.py`
3. Add tests in `tests/integration/test_faiss_retrieval.py`
4. Update `main.py` to support `--retrieval_method faiss`
5. Document in README.md and REPRO_PLAN.md

**Adding a new evaluation metric:**

1. Add function to `utils/metrics.py` with type hints and docstring
2. Add tests in `tests/unit/test_metrics.py`
3. Update `utils/evaluator.py` to compute new metric
4. Update output schema in `generate_results_table.py`

## 🔬 Reproducibility Guidelines

Since this is a **reproducibility project**, maintain:

1. **Exact version pinning**: Update `requirements.txt` with `==` versions
2. **Deterministic seeding**: Set seeds for random, numpy, torch, CUDA
3. **Document hardware**: Record GPU model, driver version, CUDA version
4. **SHA256 hashes**: Compute for all datasets
5. **Hyperparameter documentation**: Update NOTES_HPARAMS.md if changing any generation parameters

## 📖 Additional Resources

- **Paper**: [Follow My Instruction and Spill the Beans (ICLR 2025)](https://openreview.net/forum?id=oT4RSsWIyL)
- **Project Documentation**: See `/docs/` directory
- **Detailed Architecture**: `REPO_MAP.md`
- **Reproduction Plan**: `REPRO_PLAN.md`
- **Hyperparameter Notes**: `NOTES_HPARAMS.md`

## 📧 Contact

For questions or discussion:
- **Open an issue** on GitHub for bugs or feature requests
- **Email maintainer**: brightliu@college.harvard.edu

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to reproducible AI safety research!** 🎯🔬
