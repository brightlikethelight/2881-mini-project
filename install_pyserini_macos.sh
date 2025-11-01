#!/bin/bash
# Workaround script to install pyserini on macOS without nmslib
# nmslib is an optional dependency that fails to compile on Apple Silicon

set -e

echo "Installing pyserini dependencies..."

# Install all the required dependencies that pyserini needs
# pip install 'anserini==0.23.0'
pip install 'lightgbm>=2.3.1'
pip install 'Cython>=0.29.14'
pip install 'pybind11>=2.12.0'

echo "Installing pyserini without dependencies (to skip nmslib)..."

# Install pyserini itself without dependencies
# This will skip nmslib which is optional
pip install --no-deps pyserini==0.23.0

echo "Verifying installation..."

# Test if pyserini works
python -c "
try:
    from pyserini.search.lucene import LuceneSearcher
    print('✓ Pyserini installed successfully (BM25/Lucene features available)')
except ImportError as e:
    print(f'✗ Import failed: {e}')
    exit(1)

try:
    import nmslib
    print('✓ nmslib available (dense retrieval)')
except ImportError:
    print('⚠ nmslib not available (dense retrieval disabled, but BM25 still works)')
"

echo "Done! Pyserini is installed (without nmslib for macOS compatibility)"
