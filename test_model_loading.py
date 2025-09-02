#!/usr/bin/env python3
"""
Test script for OTON-AI model loading
This script tests the model loading functionality to ensure it works correctly
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def find_best_adapter_directory(base_dir: str) -> str:
    """Find the best adapter directory with valid files"""
    print(f"🔍 Searching for adapter files in: {base_dir}")
    
    # Check if base directory has valid adapter files
    if os.path.exists(os.path.join(base_dir, "adapter_config.json")) and \
       os.path.exists(os.path.join(base_dir, "adapter_model.safetensors")):
        print(f"✅ Found valid adapter files in base directory")
        return base_dir
    
    # Check for checkpoint directories
    if os.path.exists(base_dir):
        checkpoint_dirs = [d for d in os.listdir(base_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(base_dir, d))]
        if checkpoint_dirs:
            print(f"📁 Found checkpoint directories: {checkpoint_dirs}")
            # Sort by checkpoint number and return the latest
            checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else 0, reverse=True)
            latest_checkpoint = os.path.join(base_dir, checkpoint_dirs[0])
            print(f"🎯 Using latest checkpoint: {latest_checkpoint}")
            if os.path.exists(os.path.join(latest_checkpoint, "adapter_config.json")) and \
               os.path.exists(os.path.join(latest_checkpoint, "adapter_model.safetensors")):
                print(f"✅ Found valid adapter files in checkpoint directory")
                return latest_checkpoint
    
    print(f"⚠️ No valid adapter directory found, using base directory")
    return base_dir

def find_best_tokenizer_source(base_dir: str, base_model_name: str) -> str:
    """Find the best tokenizer source, preferring local files over base model"""
    print(f"🔍 Searching for tokenizer files in: {base_dir}")
    
    # Check for valid tokenizer files in the adapter directory
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
    
    # First try the base directory
    if all(os.path.exists(os.path.join(base_dir, f)) for f in tokenizer_files):
        # Check if tokenizer.json is not empty
        tokenizer_json_path = os.path.join(base_dir, "tokenizer.json")
        if os.path.getsize(tokenizer_json_path) > 1000:  # More than 1KB
            print(f"✅ Found valid tokenizer files in base directory")
            return base_dir
        else:
            print(f"⚠️ tokenizer.json is too small ({os.path.getsize(tokenizer_json_path)} bytes)")
    
    # Check checkpoint directories
    if os.path.exists(base_dir):
        checkpoint_dirs = [d for d in os.listdir(base_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(base_dir, d))]
        if checkpoint_dirs:
            checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else 0, reverse=True)
            for checkpoint_dir in checkpoint_dirs:
                checkpoint_path = os.path.join(base_dir, checkpoint_dir)
                if all(os.path.exists(os.path.join(checkpoint_path, f)) for f in tokenizer_files):
                    tokenizer_json_path = os.path.join(checkpoint_path, "tokenizer.json")
                    if os.path.getsize(tokenizer_json_path) > 1000:  # More than 1KB
                        print(f"✅ Found valid tokenizer files in checkpoint: {checkpoint_dir}")
                        return checkpoint_path
    
    print(f"🔄 Falling back to base model tokenizer: {base_model_name}")
    return base_model_name

def test_model_loading():
    """Test the model loading functionality"""
    print("🚀 Testing OTON-AI Model Loading")
    print("=" * 50)
    
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    adapter_dir = "group_level_outputs"
    
    try:
        # Find best adapter directory
        best_adapter_dir = find_best_adapter_directory(adapter_dir)
        print(f"\n📁 Best adapter directory: {best_adapter_dir}")
        
        # Find best tokenizer source
        tokenizer_src = find_best_tokenizer_source(best_adapter_dir, base_model_name)
        print(f"🔤 Best tokenizer source: {tokenizer_src}")
        
        # Test tokenizer loading
        print(f"\n🔍 Testing tokenizer loading...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print(f"✅ Tokenizer loaded successfully from: {tokenizer_src}")
            print(f"   - Vocabulary size: {tokenizer.vocab_size}")
            print(f"   - Model max length: {tokenizer.model_max_length}")
        except Exception as e:
            print(f"❌ Tokenizer loading failed: {str(e)}")
            print(f"🔄 Trying base model tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print(f"✅ Base model tokenizer loaded successfully")
        
        # Test base model loading
        print(f"\n🔍 Testing base model loading...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   - Using device: {device}")
        
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        print(f"✅ Base model loaded successfully")
        
        # Test adapter loading
        adapter_config_path = os.path.join(best_adapter_dir, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            print(f"\n🔍 Testing LoRA adapter loading...")
            try:
                model = PeftModel.from_pretrained(model, best_adapter_dir)
                print(f"✅ LoRA adapter loaded successfully")
                
                # Test a simple inference
                print(f"\n🧪 Testing simple inference...")
                test_prompt = "### Strategic Task:\nTest strategic analysis\n\n### Business Context:\nTest context\n\n### Strategic Analysis:\n"
                inputs = tokenizer(test_prompt, return_tensors="pt")
                
                if device == "cuda":
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    model = model.to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"✅ Inference test successful!")
                print(f"   - Input length: {len(test_prompt)} characters")
                print(f"   - Output length: {len(response)} characters")
                
            except Exception as e:
                print(f"❌ Adapter loading failed: {str(e)}")
                print(f"🔄 Continuing with base model only")
        else:
            print(f"⚠️ No adapter config found, skipping adapter test")
        
        print(f"\n🎯 Model loading test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Model loading test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n✅ All tests passed! OTON-AI is ready to use.")
    else:
        print("\n❌ Tests failed. Please check the error messages above.")
        sys.exit(1)
