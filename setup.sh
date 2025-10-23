#!/bin/bash
# One-shot setup script for RAG copy-out experiment
# For ICLR 2025: "Follow My Instruction and Spill the Beans"

set -e  # Exit on error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_msg() {
    local color=$1
    shift
    echo -e "${color}$*${NC}"
}

print_header() {
    echo ""
    print_msg "$BLUE" "======================================"
    print_msg "$BLUE" "$1"
    print_msg "$BLUE" "======================================"
    echo ""
}

print_header "RAG Privacy - Setup Script"
print_msg "$GREEN" "For ICLR 2025: Follow My Instruction and Spill the Beans"
echo ""

# Step 1: Check prerequisites
print_header "[1/7] Checking Prerequisites"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    print_msg "$RED" "✗ Error: Conda not found"
    echo ""
    print_msg "$YELLOW" "Please install Anaconda or Miniconda first:"
    print_msg "$YELLOW" "  https://docs.conda.io/en/latest/miniconda.html"
    echo ""
    exit 1
fi
print_msg "$GREEN" "✓ Conda found: $(conda --version)"

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
print_msg "$GREEN" "✓ Python found: $(python --version)"

# Step 2: Check if environment already exists
print_header "[2/7] Checking Environment"

ENV_NAME="rag-privacy"
if conda env list | grep -q "^${ENV_NAME} "; then
    print_msg "$YELLOW" "⚠ Environment '$ENV_NAME' already exists"
    read -p "  Do you want to remove and recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_msg "$YELLOW" "  Removing existing environment..."
        conda env remove -n "$ENV_NAME" -y
    else
        print_msg "$YELLOW" "  Using existing environment"
        conda activate "$ENV_NAME" 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$ENV_NAME"
        print_msg "$GREEN" "✓ Environment activated"

        # Skip environment creation
        SKIP_ENV_CREATION=true
    fi
fi

# Step 3: Create conda environment
if [ "$SKIP_ENV_CREATION" != "true" ]; then
    print_header "[3/7] Creating Conda Environment"

    if [ -f "environment.yml" ]; then
        print_msg "$BLUE" "Using environment.yml..."
        conda env create -f environment.yml
    else
        print_msg "$BLUE" "Creating environment manually..."
        conda create -n "$ENV_NAME" python=3.10 -y
        conda activate "$ENV_NAME" || source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$ENV_NAME"

        # Install system dependencies
        print_msg "$BLUE" "Installing Java and FAISS..."
        conda install -c conda-forge openjdk=21 -y
        conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 -y 2>/dev/null || \
            conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 -y
    fi

    print_msg "$GREEN" "✓ Environment created: $ENV_NAME"
fi

# Activate environment
print_msg "$BLUE" "Activating environment..."
conda activate "$ENV_NAME" 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$ENV_NAME"

# Step 4: Verify Java
print_header "[4/7] Verifying Java Installation"

if ! command -v java &> /dev/null; then
    print_msg "$RED" "✗ Error: Java not found"
    print_msg "$YELLOW" "Installing Java via conda..."
    conda install -c conda-forge openjdk=21 -y
fi

JAVA_VERSION=$(java -version 2>&1 | head -1)
print_msg "$GREEN" "✓ Java found: $JAVA_VERSION"

# Set LD_LIBRARY_PATH for Linux (fixes GLIBCXX errors)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
    print_msg "$GREEN" "✓ Set LD_LIBRARY_PATH for Linux"

    # Add to bashrc if not already there
    BASHRC="$HOME/.bashrc"
    LD_LINE="export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
    if ! grep -q "$LD_LINE" "$BASHRC" 2>/dev/null; then
        print_msg "$YELLOW" "  Adding LD_LIBRARY_PATH to ~/.bashrc for future sessions"
        echo "" >> "$BASHRC"
        echo "# Added by RAG privacy setup.sh" >> "$BASHRC"
        echo "$LD_LINE" >> "$BASHRC"
    fi
fi

# Step 5: Install PyTorch
print_header "[5/7] Installing PyTorch"

# Detect CUDA
HAS_CUDA=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        HAS_CUDA=true
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        print_msg "$GREEN" "✓ GPU detected: $GPU_INFO"
    fi
fi

if [ "$HAS_CUDA" = true ]; then
    print_msg "$BLUE" "Installing PyTorch with CUDA 12.1..."
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cu121
else
    print_msg "$YELLOW" "No GPU detected, installing CPU-only PyTorch..."
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cpu
fi

print_msg "$GREEN" "✓ PyTorch installed"

# Step 6: Install Python dependencies
print_header "[6/7] Installing Python Dependencies"

# On macOS, use special requirements file without nmslib
if [[ "$OSTYPE" == "darwin"* ]] && [ -f "requirements-macos.txt" ]; then
    print_msg "$YELLOW" "⚠ macOS detected: Using macOS-specific requirements (without nmslib)"
    print_msg "$BLUE" "Installing from requirements-macos.txt..."
    pip install -r requirements-macos.txt

    # Install pyserini core components without optional dependencies
    print_msg "$BLUE" "Installing pyserini (core only, without nmslib)..."
    print_msg "$YELLOW" "  Note: nmslib will be skipped (it's optional and incompatible with Apple Silicon)"

    # Install pyserini's known dependencies manually first
    print_msg "$BLUE" "  Installing pyserini dependencies..."
    pip install 'lightgbm>=2.3.1' 'Cython>=0.29.14' 'pybind11>=2.5.0' 'anserini==0.23.0' || true

    # Now try to install pyserini, allowing nmslib to fail
    print_msg "$BLUE" "  Attempting pyserini installation (nmslib failures are expected and OK)..."

    # We'll continue even if nmslib fails
    pip install --no-deps pyserini==0.23.0 2>&1 | grep -v "nmslib" || true

    # Verify pyserini is importable
    if python -c "import pyserini" 2>/dev/null; then
        print_msg "$GREEN" "✓ pyserini installed successfully (without nmslib)"
    else
        print_msg "$RED" "✗ pyserini installation failed"
        print_msg "$YELLOW" "  This is a known issue on macOS. You may need to:"
        print_msg "$YELLOW" "  1. Skip pyserini and use only kNN-LM experiments"
        print_msg "$YELLOW" "  2. Or manually compile nmslib with: CFLAGS=\"-mavx -DWARN(a)=(a)\" pip install nmslib"
    fi
    print_msg "$GREEN" "✓ Core dependencies installed (nmslib skipped on macOS)"

elif [ -f "requirements.txt" ]; then
    print_msg "$BLUE" "Installing from requirements.txt..."
    pip install -r requirements.txt
    print_msg "$GREEN" "✓ Core dependencies installed"
else
    print_msg "$YELLOW" "⚠ requirements.txt not found, installing manually..."
    pip install transformers datasets accelerate \
                pyserini \
                evaluate rouge-score sacrebleu bert-score nltk \
                scipy numpy tqdm wandb
    print_msg "$GREEN" "✓ Dependencies installed"
fi

# Download NLTK data
print_msg "$BLUE" "Downloading NLTK punkt tokenizer..."
python -c "import nltk; nltk.download('punkt', quiet=True)" 2>/dev/null || true
print_msg "$GREEN" "✓ NLTK data downloaded"

# Step 7: Verify installation
print_header "[7/7] Verifying Installation"

python << 'EOF'
import sys
import warnings
warnings.filterwarnings("ignore")

print("Python:", sys.version.split()[0])

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
except ImportError as e:
    print(f"✗ PyTorch: {e}")
    sys.exit(1)

try:
    import transformers
    print(f"✓ Transformers: {transformers.__version__}")
except ImportError as e:
    print(f"✗ Transformers: {e}")
    sys.exit(1)

try:
    from pyserini.search.lucene import LuceneSearcher
    print(f"✓ Pyserini: OK")
except ImportError as e:
    print(f"✗ Pyserini: {e}")
    sys.exit(1)

try:
    import faiss
    print(f"✓ FAISS: {faiss.__version__ if hasattr(faiss, '__version__') else 'OK'}")
except ImportError as e:
    print(f"✗ FAISS: {e}")
    sys.exit(1)

try:
    import evaluate
    print(f"✓ Evaluate: {evaluate.__version__}")
except ImportError as e:
    print(f"✗ Evaluate: {e}")
    sys.exit(1)

print("\n✅ All dependencies installed successfully!")
EOF

if [ $? -ne 0 ]; then
    print_msg "$RED" "\n✗ Installation verification failed"
    print_msg "$YELLOW" "Please check error messages above"
    exit 1
fi

# Final success message
print_header "Installation Complete!"

print_msg "$GREEN" "✓ Environment: $ENV_NAME"
print_msg "$GREEN" "✓ Python: $(python --version | awk '{print $2}')"
print_msg "$GREEN" "✓ PyTorch: $(python -c 'import torch; print(torch.__version__)')"
if [ "$HAS_CUDA" = true ]; then
    print_msg "$GREEN" "✓ CUDA: Available"
else
    print_msg "$YELLOW" "⚠ CUDA: Not available (CPU mode)"
fi

echo ""
print_msg "$BLUE" "Next steps:"
print_msg "$BLUE" "  1. Activate environment:"
print_msg "$BLUE" "       conda activate $ENV_NAME"
echo ""
print_msg "$BLUE" "  2. Fetch Wikipedia data:"
print_msg "$BLUE" "       python scripts/fetch_wikipedia.py --help"
echo ""
print_msg "$BLUE" "  3. Generate attack prompts:"
print_msg "$BLUE" "       python scripts/generate_prompts.py"
echo ""
print_msg "$BLUE" "  4. Run experiments:"
print_msg "$BLUE" "       python main.py --task io ..."
echo ""
print_msg "$BLUE" "  5. See REPRO_PLAN.md for complete workflow"
echo ""

print_msg "$GREEN" "Happy reproducing! 🚀"
echo ""
