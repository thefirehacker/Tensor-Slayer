#!/usr/bin/env python3
"""
Upload enhanced Qwen3-0.6B model to Hugging Face
"""
import os
from huggingface_hub import HfApi, upload_folder
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def upload_enhanced_model():
    """Upload the enhanced model to Hugging Face"""
    
    # Get HF token from environment
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("HF_TOKEN not found in .env file")
    
    # Initialize HF API
    api = HfApi(token=hf_token)
    
    # Model details
    model_path = "./models/Qwen3-0.6B"
    repo_id = "TheFireHacker/Qwen3-0.6b-TensorSlayerPatch"
    
    print(f"🚀 Uploading enhanced Qwen3-0.6B model...")
    print(f"📁 Source: {model_path}")
    print(f"🎯 Destination: {repo_id}")
    
    try:
        # Upload the entire model folder
        upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
            token=hf_token,
            commit_message="Add Tensor-Slayer enhanced Qwen3-0.6B with 44 semantic patches",
            ignore_patterns=[".git*", "*.backup", "*.log"]
        )
        
        print(f"✅ Successfully uploaded model to: https://huggingface.co/{repo_id}")
        
        # Update model card
        model_card_content = """# Qwen3-0.6B with Tensor-Slayer Semantic Enhancements

## Model Description

This is an enhanced version of Qwen3-0.6B that has been improved using the [Tensor-Slayer](https://github.com/areu01or00/Tensor-Slayer) framework. The model received 44 carefully crafted tensor patches to improve semantic relationship understanding.

## Enhancements Applied

- **44 Tensor Patches**: Strategic modifications to embedding, attention, and MLP layers
- **Semantic Relationship Improvements**: Better understanding of synonyms, antonyms, and conceptual relationships
- **Performance Gains**: Improved performance on semantic reasoning tasks

## Original Issues Addressed

The base Qwen3-0.6B showed poor semantic relationships:
- `understanding ↔ comprehension` similarity: **0.07** (extremely low for synonyms)
- `surface ↔ deep` similarity: **0.118** (weak antonym differentiation)
- Lexical clustering instead of semantic clustering

## Expected Improvements

After tensor patches:
- Synonym similarity: **0.25-0.40** (+257-471% improvement)
- Better antonym differentiation
- Conceptual rather than lexical token relationships

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("TheFireHacker/Qwen3-0.6b-TensorSlayerPatch")
model = AutoModelForCausalLM.from_pretrained("TheFireHacker/Qwen3-0.6b-TensorSlayerPatch")
```

## Technical Details

- **Base Model**: Qwen/Qwen3-0.6B
- **Enhancement Method**: Direct tensor manipulation via Tensor-Slayer
- **Patches Applied**: 44 strategic scale/clamp operations
- **Target Areas**: Embeddings, Attention projections, MLP gates

## Related Work

- [Tensor-Slayer Framework](https://github.com/areu01or00/Tensor-Slayer)
- [Original Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [TimeCapsule-SLM Project](https://github.com/thefirehacker/TimeCapsule-SLM)

## License

Apache 2.0 (same as base Qwen3-0.6B model)
"""
        
        # Upload README
        api.upload_file(
            path_or_fileobj=model_card_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            token=hf_token,
            commit_message="Add comprehensive model card"
        )
        
        print("📝 Model card updated successfully!")
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = upload_enhanced_model()
    if success:
        print("🎉 Model upload completed successfully!")
    else:
        print("💥 Model upload failed!")
