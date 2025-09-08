#!/bin/bash
# Quick script to upload GGUF models to Ollama

echo "🚀 Uploading Enhanced Qwen3-0.6B Models to Ollama"
echo "=================================================="

# Check if Ollama is running
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found. Please install Ollama first."
    exit 1
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please create it with ollama_api variable."
    exit 1
fi

# Install required dependencies
echo "📦 Installing Python dependencies..."
pip install -q requests python-dotenv

# Run the upload script
echo "🎯 Starting upload process..."
python upload_to_ollama.py

echo "✅ Upload script completed!"
echo ""
echo "🧪 Test your models:"
echo "ollama run qwen3-enhanced-fast 'Compare understanding and comprehension'"
echo "ollama run qwen3-enhanced-balanced 'Explain semantic relationships'"
echo "ollama run qwen3-enhanced-quality 'Rate similarity: surface vs deep'"
