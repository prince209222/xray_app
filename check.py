import os
import sys
import subprocess

def verify():
    print("🔍 Running pre-flight checks...")
    
    # 1. Python version
    py_ok = (3, 9) <= sys.version_info < (3, 10)
    print(f"Python 3.9.x: {'✅' if py_ok else '❌'} (Found: {sys.version})")
    
    # 2. Model file
    model_exists = os.path.exists('models/medical_resnet34.pt')
    print(f"Model file: {'✅' if model_exists else '❌'} at models/medical_resnet34.pt")
    
    # 3. Dependencies
    try:
        import torch
        torch_ok = torch.__version__.startswith('2.1.0')
        print(f"PyTorch 2.1.0: {'✅' if torch_ok else '❌'} (Found: {torch.__version__})")
    except ImportError:
        print("PyTorch: ❌ Not installed")
    
    if all([py_ok, model_exists]):
        print("\n✅ System ready for deployment!")
    else:
        print("\n❌ Fix issues before deployment")

if __name__ == "__main__":
    verify()
