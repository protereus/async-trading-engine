#!/bin/bash
set -e

echo "Building llama.cpp..."
cd llama.cpp
cmake -B build
cmake --build build -j 1

echo "Downloading the Strand-Rust-Coder model..."
python3 -m venv hf-env
source hf-env/bin/activate
pip install -U "huggingface_hub[cli]"
hf download mradermacher/Strand-Rust-Coder-14B-v1-i1-GGUF --include "*IQ3_XXS.gguf" --local-dir .
deactivate
cd ..

echo "Installing NVM, Node.js, and OpenCode..."
wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.8/install.sh | bash

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

nvm install 20
npm install -g opencode-ai

echo "Setup fully complete."
