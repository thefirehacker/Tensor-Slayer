#!/usr/bin/env python3
"""
Upload Enhanced Qwen3-0.6B GGUF Models to Ollama
Downloads GGUF models from HuggingFace and creates Ollama models
"""

import os
import requests
import subprocess
import time
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class OllamaUploader:
    def __init__(self):
        self.ollama_api_key = os.getenv('ollama_api')
        if not self.ollama_api_key:
            raise ValueError("ollama_api key not found in .env file")
        
        # Ollama's cloud API endpoint
        self.ollama_base_url = "https://api.ollama.ai"
        self.headers = {
            "Authorization": f"Bearer {self.ollama_api_key}",
            "Content-Type": "application/json"
        }
        
        self.models_dir = Path("./gguf_models")
        self.models_dir.mkdir(exist_ok=True)
        
        # GGUF model configurations
        self.models = {
            "qwen3-enhanced-fast": {
                "file": "qwen3-0.6b-tensorslayer-q4_0.gguf",
                "url": "https://huggingface.co/TheFireHacker/Qwen3-0.6b-TensorSlayerPatch/resolve/main/gguf/qwen3-0.6b-tensorslayer-q4_0.gguf",
                "size": "382MB",
                "description": "Q4_0 - Fastest inference, good quality"
            },
            "qwen3-enhanced-balanced": {
                "file": "qwen3-0.6b-tensorslayer-q5_k_m.gguf", 
                "url": "https://huggingface.co/TheFireHacker/Qwen3-0.6b-TensorSlayerPatch/resolve/main/gguf/qwen3-0.6b-tensorslayer-q5_k_m.gguf",
                "size": "444MB",
                "description": "Q5_K_M - Balanced speed and quality"
            },
            "qwen3-enhanced-quality": {
                "file": "qwen3-0.6b-tensorslayer-q8_0.gguf",
                "url": "https://huggingface.co/TheFireHacker/Qwen3-0.6b-TensorSlayerPatch/resolve/main/gguf/qwen3-0.6b-tensorslayer-q8_0.gguf", 
                "size": "639MB",
                "description": "Q8_0 - High quality, slower inference"
            },
            "qwen3-enhanced-max": {
                "file": "qwen3-0.6b-tensorslayer-f16.gguf",
                "url": "https://huggingface.co/TheFireHacker/Qwen3-0.6b-TensorSlayerPatch/resolve/main/gguf/qwen3-0.6b-tensorslayer-f16.gguf",
                "size": "1.2GB", 
                "description": "F16 - Maximum quality, full precision"
            }
        }
        
        print(f"🔗 Ollama API: {self.ollama_base_url}")
        print(f"🔑 API Key: {self.ollama_api_key[:8]}...{self.ollama_api_key[-4:]}")
        print(f"📁 Models directory: {self.models_dir}")

    def check_ollama_connection(self):
        """Test connection to Ollama API"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/models", headers=self.headers, timeout=10)
            if response.status_code == 200:
                print("✅ Ollama API connection successful!")
                return True
            else:
                print(f"❌ Ollama API responded with status: {response.status_code}")
                if response.status_code == 401:
                    print("💡 Check your ollama_api key in .env file")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to Ollama API: {e}")
            print("💡 Check your ollama_api key and internet connection")
            return False

    def download_model(self, model_name, model_info):
        """Download GGUF model from HuggingFace"""
        file_path = self.models_dir / model_info["file"]
        
        # Skip if already exists
        if file_path.exists():
            print(f"✅ {model_info['file']} already exists ({model_info['size']})")
            return file_path
        
        print(f"📥 Downloading {model_name} ({model_info['size']})...")
        print(f"🔗 URL: {model_info['url']}")
        
        try:
            # Download with progress
            response = requests.get(model_info["url"], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Progress indicator
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r📊 Progress: {percent:.1f}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)", end="")
            
            print(f"\n✅ Downloaded {model_info['file']}")
            return file_path
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            if file_path.exists():
                file_path.unlink()  # Remove partial file
            return None

    def upload_ollama_model(self, model_name, model_info, file_path):
        """Upload GGUF model to Ollama cloud service"""
        
        print(f"🚀 Uploading {model_name} to Ollama cloud...")
        
        try:
            # Step 1: Create model metadata
            model_data = {
                "name": model_name,
                "description": f"Tensor-Slayer Enhanced Qwen3-0.6B - {model_info['description']}",
                "template": """<|im_start|>system
You are a helpful assistant with enhanced semantic understanding.
Enhanced with 44 tensor patches for improved semantic relationships.<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
""",
                "parameters": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stop": ["<|im_start|>", "<|im_end|>"]
                }
            }
            
            # Step 2: Upload GGUF file
            with open(file_path, 'rb') as f:
                files = {
                    'file': (file_path.name, f, 'application/octet-stream'),
                    'metadata': (None, requests.utils.json.dumps(model_data), 'application/json')
                }
                
                # Remove Content-Type header for multipart upload
                upload_headers = {"Authorization": f"Bearer {self.ollama_api_key}"}
                
                print(f"📤 Uploading {file_path.name} ({model_info['size']})...")
                response = requests.post(
                    f"{self.ollama_base_url}/api/models/upload",
                    files=files,
                    headers=upload_headers,
                    timeout=600  # 10 minutes timeout for large files
                )
            
            if response.status_code == 200 or response.status_code == 201:
                print(f"✅ Successfully uploaded {model_name} to Ollama!")
                return True
            else:
                print(f"❌ Upload failed with status {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Upload error: {e}")
            return False

    def test_model(self, model_name):
        """Test the uploaded model"""
        test_prompt = "Rate the similarity between 'understanding' and 'comprehension' on a scale of 1-10 and explain why."
        
        try:
            print(f"🧪 Testing {model_name}...")
            
            # Test via Ollama API
            response = requests.post(f"{self.ollama_base_url}/api/generate", 
                json={
                    "model": model_name,
                    "prompt": test_prompt,
                    "stream": False
                }, 
                headers=self.headers,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', 'No response')[:200] + "..."
                print(f"✅ {model_name} test successful!")
                print(f"🤖 Response preview: {answer}")
                return True
            else:
                print(f"❌ Test failed with status: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Test error: {e}")
            return False

    def list_ollama_models(self):
        """List current Ollama models"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/models", headers=self.headers)
            if response.status_code == 200:
                models = response.json().get('models', [])
                print(f"\n📋 Current Ollama models ({len(models)}):")
                for model in models:
                    name = model.get('name', 'Unknown')
                    size = model.get('size', 0) // (1024*1024)  # Convert to MB
                    print(f"  • {name} ({size}MB)")
                return models
            else:
                print(f"❌ Failed to list models: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Error listing models: {e}")
            return []

    def upload_all_models(self, skip_existing=True):
        """Upload all GGUF models to Ollama"""
        
        print("🚀 Starting Ollama upload process...")
        print("=" * 60)
        
        # Check connection first
        if not self.check_ollama_connection():
            return False
        
        # Show current models
        self.list_ollama_models()
        
        successful_uploads = []
        failed_uploads = []
        
        for model_name, model_info in self.models.items():
            print(f"\n📦 Processing {model_name}...")
            print(f"📊 {model_info['description']} ({model_info['size']})")
            
            # Download model
            file_path = self.download_model(model_name, model_info)
            if not file_path:
                failed_uploads.append(model_name)
                continue
            
            # Upload to Ollama
            if self.upload_ollama_model(model_name, model_info, file_path):
                # Test model
                if self.test_model(model_name):
                    successful_uploads.append(model_name)
                else:
                    failed_uploads.append(model_name)
            else:
                failed_uploads.append(model_name)
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 UPLOAD SUMMARY")
        print("=" * 60)
        
        if successful_uploads:
            print(f"✅ Successfully uploaded ({len(successful_uploads)}):")
            for model in successful_uploads:
                info = self.models[model]
                print(f"  • {model} - {info['description']} ({info['size']})")
        
        if failed_uploads:
            print(f"\n❌ Failed uploads ({len(failed_uploads)}):")
            for model in failed_uploads:
                print(f"  • {model}")
        
        print(f"\n🎯 Recommended model for production: qwen3-enhanced-balanced")
        print(f"🧪 Test with: ollama run qwen3-enhanced-balanced 'Compare understanding and comprehension'")
        
        return len(successful_uploads) > 0

def main():
    """Main execution function"""
    
    print("🎯 Qwen3-0.6B TensorSlayer Enhanced - Ollama Upload")
    print("=" * 60)
    
    try:
        uploader = OllamaUploader()
        success = uploader.upload_all_models()
        
        if success:
            print("\n🎉 Upload process completed successfully!")
            print("🔗 Models available via Ollama API and CLI")
        else:
            print("\n💥 Upload process failed!")
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return False
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
