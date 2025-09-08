# Google Colab GGUF Conversion Instructions

## Setup Steps for Google Colab A100

### 1. Open Google Colab
- Go to [Google Colab](https://colab.research.google.com/)
- Create a new notebook
- Set runtime to **GPU (A100 recommended)**

### 2. Copy the Conversion Script

Copy the entire content from `colab_gguf_conversion.py` and paste it into your Colab notebook.

### 3. Run the Conversion

#### Cell 1: Install Dependencies
```python
# Install required packages
!pip install huggingface_hub transformers torch safetensors accelerate
!pip install gguf

# Clone llama.cpp for conversion tools
!git clone https://github.com/ggerganov/llama.cpp.git
!cd llama.cpp && make -j$(nproc)
```

#### Cell 2: Setup and Login
```python
import os
import torch
from huggingface_hub import HfApi, upload_folder, login
from transformers import AutoTokenizer, AutoModelForCausalLM
import subprocess
import shutil

# Login to Hugging Face (enter your token when prompted)
login()

# Verify GPU
print(f"🚀 GPU Available: {torch.cuda.is_available()}")
print(f"🔧 GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

#### Cell 3: Run Main Conversion
```python
# Paste the main() function and all helper functions here
# Then run:
main()
```

### 4. Expected Output

The script will:
1. ✅ Download your enhanced model from HuggingFace
2. ✅ Convert to multiple GGUF formats:
   - **FP16** (~1.2GB) - Maximum quality
   - **Q8_0** (~650MB) - High quality, smaller
   - **Q5_K_M** (~450MB) - Balanced quality/size
   - **Q4_0** (~350MB) - Fastest inference
3. ✅ Create proper folder structure
4. ✅ Upload to `gguf/` folder in your repo

### 5. Verification

After completion, check:
- [https://huggingface.co/TheFireHacker/Qwen3-0.6b-TensorSlayerPatch/tree/main/gguf](https://huggingface.co/TheFireHacker/Qwen3-0.6b-TensorSlayerPatch/tree/main/gguf)

You should see:
```
gguf/
├── README.md
├── qwen3-0.6b-tensorslayer-f16.gguf
├── qwen3-0.6b-tensorslayer-q8_0.gguf
├── qwen3-0.6b-tensorslayer-q5_k_m.gguf
└── qwen3-0.6b-tensorslayer-q4_0.gguf
```

## Testing with Ollama (After Conversion)

Once GGUF files are uploaded, test locally:

```bash
# Download the Q4_0 version (fastest)
wget https://huggingface.co/TheFireHacker/Qwen3-0.6b-TensorSlayerPatch/resolve/main/gguf/qwen3-0.6b-tensorslayer-q4_0.gguf

# Create Ollama model
ollama create qwen3-enhanced -f - <<EOF
FROM ./qwen3-0.6b-tensorslayer-q4_0.gguf
TEMPLATE """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOF

# Test enhanced semantic relationships
ollama run qwen3-enhanced "Rate the similarity between 'understanding' and 'comprehension' on a scale of 1-10 and explain why."
```

## Expected Semantic Improvements

After tensor patches, you should see:
- **Better synonym recognition** (understanding ↔ comprehension)
- **Improved antonym differentiation** (surface ↔ deep)
- **Conceptual clustering** instead of lexical clustering
- **Enhanced abstract reasoning**

## Troubleshooting

### Common Issues:
1. **CUDA out of memory**: Restart runtime and try again
2. **HuggingFace login fails**: Check your token permissions
3. **Conversion errors**: Ensure llama.cpp built successfully
4. **Upload fails**: Check internet connection and HF token

### Alternative Quantizations:
If you need different quantizations, modify the `conversion_commands` list in the script.

## Performance Expectations

| Format | Size | Speed | Quality | Best For |
|--------|------|-------|---------|----------|
| F16 | 1.2GB | Slow | Best | Research/Analysis |
| Q8_0 | 650MB | Medium | High | Production |
| Q5_K_M | 450MB | Fast | Good | Balanced |
| Q4_0 | 350MB | Fastest | Acceptable | Edge devices |

Choose based on your hardware and quality requirements.
