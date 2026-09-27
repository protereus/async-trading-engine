#!/bin/bash
set -e

apt update && apt install -y build-essential cmake git wget python3-venv curl

if [ ! -f "llama.cpp/build/bin/llama-server" ]; then
    echo "Building llama.cpp..."
    if [ ! -d "llama.cpp" ]; then
        git clone https://github.com/ggerganov/llama.cpp
    fi
    cd llama.cpp
    cmake -B build
    cmake --build build -j 4
    cd ..
fi

if [ ! -f "Strand-Rust-Coder-14B-v1.i1-IQ3_XXS.gguf" ]; then
    echo "Downloading model directly..."
    wget -c https://huggingface.co/mradermacher/Strand-Rust-Coder-14B-v1-i1-GGUF/resolve/main/Strand-Rust-Coder-14B-v1.i1-IQ3_XXS.gguf
fi

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
fi

echo "Setup fully complete."
