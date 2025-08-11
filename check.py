import os
import sys
import subprocess

def verify():
    print("🔍 Running verification...")
    
    # 1. Python version
    assert (3, 9) <= sys.version_info < (3, 10), f"Python 3.9.x required (found {sys.version})"
    
    # 2. Model file
    model_path = 'models/medical_resnet34.pt'
    assert os.path.exists(model_path), f"Model not found at {os.path.abspath(model_path)}"
    
    # 3. Dependencies
    try:
        import torch
        assert torch.__version__ == '2.1.0', f"Torch 2.1.0 required (found {torch.__version__})"
        print("✅ All checks passed!")
    except Exception as e:
        print(f"❌ Verification failed: {str(e)}")
        print("Install dependencies with:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    verify()