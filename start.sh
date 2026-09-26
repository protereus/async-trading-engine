#!/bin/bash
set -e

# 1. Skip compilation if it is already finished
if [ ! -f "llama.cpp/build/bin/llama-server" ]; then
    echo "Building llama.cpp..."
    cd llama.cpp
    cmake -B build
    cmake --build build -j 1
    cd ..
else
    echo "✓ llama.cpp already built. Skipping..."
fi

# 2. Resumable model download
if ! ls *IQ3_XXS.gguf 1> /dev/null 2>&1; then
    echo "Downloading model (Resumes automatically if interrupted)..."
    if [ ! -d "hf-env" ]; then
        python3 -m venv hf-env
    fi
    source hf-env/bin/activate
    pip install -q -U "huggingface_hub[cli]"
    hf download mradermacher/Strand-Rust-Coder-14B-v1-i1-GGUF --include "*IQ3_XXS.gguf" --local-dir .
    deactivate
else
    echo "✓ Model already downloaded. Skipping..."
fi

# 3. Skip installation if OpenCode is already present
export NVM_DIR="$HOME/.nvm"
if [ ! -d "$NVM_DIR" ]; then
    echo "Installing NVM..."
    wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.8/install.sh | bash
fi

[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

if ! command -v opencode >/dev/null 2>&1; then
    echo "Installing Node.js and OpenCode..."
    nvm install 20
    npm install -g opencode-ai
else
    echo "✓ OpenCode already installed. Skipping..."
fi

echo "Setup fully complete."
