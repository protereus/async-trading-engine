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

# 2. Low-memory resumable download via wget
if [ ! -f "Strand-Rust-Coder-14B-v1-i1-IQ3_XXS.gguf" ]; then
    echo "Downloading model directly (low-memory mode)..."
    wget -c https://huggingface.co/mradermacher/Strand-Rust-Coder-14B-v1-i1-GGUF/resolve/main/Strand-Rust-Coder-14B-v1-i1-IQ3_XXS.gguf
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
