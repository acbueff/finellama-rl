#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive test script for verifying TRL and flash-attention functionality
in an Apptainer container environment for HPC use.
"""

import os
import sys
import argparse
import traceback

def test_torch():
    """Test PyTorch and CUDA availability."""
    print("\n=== Testing PyTorch and CUDA ===")
    import torch
    
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA is not available!")
        return False
    
    # Test tensor creation on GPU
    try:
        x = torch.tensor([1, 2, 3], device="cuda")
        print(f"Test tensor created on GPU: {x}")
        return True
    except Exception as e:
        print(f"ERROR creating tensor on GPU: {e}")
        return False

def test_flash_attention():
    """Test flash-attention functionality."""
    print("\n=== Testing Flash-Attention ===")
    try:
        import flash_attn
        from flash_attn import flash_attn_func
        print(f"Flash-attention version: {flash_attn.__version__}")
        print("Flash-attention module imported successfully")
        
        # More detailed tests could be added here, but this requires 
        # significant GPU memory and proper tensor shapes
        return True
    except ImportError:
        print("ERROR: Failed to import flash-attention module")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"ERROR with flash-attention: {e}")
        traceback.print_exc()
        return False

def test_transformers():
    """Test transformers library."""
    print("\n=== Testing Transformers ===")
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
        
        # Test tokenizer loading
        print("Testing tokenizer loading...")
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        print("Tokenizer loaded successfully")
        
        return True
    except Exception as e:
        print(f"ERROR with transformers: {e}")
        traceback.print_exc()
        return False

def test_trl():
    """Test TRL installation and GRPOTrainer."""
    print("\n=== Testing TRL ===")
    try:
        import trl
        print(f"TRL version: {trl.__version__}")
        
        # Verify GRPOTrainer class is available
        from trl import GRPOConfig, GRPOTrainer
        print("GRPOTrainer class imported successfully")
        
        # Create minimal config
        config = GRPOConfig(
            output_dir="./test_output",
            per_device_train_batch_size=1,
            max_length=128
        )
        print("GRPOConfig created successfully")
        
        # More extensive tests would create an actual trainer instance,
        # but this requires downloading models and more GPU memory
        return True
    except ImportError:
        print("ERROR: Failed to import TRL module")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"ERROR with TRL: {e}")
        traceback.print_exc()
        return False

def get_gpu_info():
    """Get GPU information using NVIDIA tools."""
    print("\n=== GPU Information ===")
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"Error getting GPU info: {e}")

def run_all_tests():
    """Run all test functions and summarize results."""
    print("\n===== Container Verification Tests =====")
    print(f"Date/Time: {os.popen('date').read().strip()}")
    print(f"Hostname: {os.popen('hostname').read().strip()}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    get_gpu_info()
    
    # Run all tests
    tests = [
        ("PyTorch/CUDA", test_torch),
        ("Flash-Attention", test_flash_attention),
        ("Transformers", test_transformers),
        ("TRL", test_trl),
    ]
    
    # Track results
    results = {}
    all_passed = True
    
    for name, test_func in tests:
        print(f"\nRunning test: {name}")
        try:
            result = test_func()
            results[name] = result
            if not result:
                all_passed = False
        except Exception as e:
            print(f"ERROR in test {name}: {e}")
            traceback.print_exc()
            results[name] = False
            all_passed = False
    
    # Print summary
    print("\n===== Test Results Summary =====")
    for name, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"{name}: {status}")
    
    if all_passed:
        print("\n✅ All tests PASSED!")
        return 0
    else:
        print("\n❌ Some tests FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
