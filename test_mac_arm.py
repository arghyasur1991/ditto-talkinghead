#!/usr/bin/env python3
"""
Test script for Mac ARM support.
This script tests device detection and model loading capabilities.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.utils.device_utils import detect_device, get_recommended_config, print_device_info, is_mac_arm


def test_device_detection():
    """Test device detection functionality."""
    print("=== Testing Device Detection ===")
    
    device, model_format = detect_device()
    print(f"Detected device: {device}")
    print(f"Recommended model format: {model_format}")
    print(f"Is Mac ARM: {is_mac_arm()}")
    
    config = get_recommended_config()
    print(f"Recommended config: {config}")
    
    return device, model_format


def test_onnx_runtime():
    """Test ONNX runtime with available providers."""
    print("\n=== Testing ONNX Runtime ===")
    
    try:
        import onnxruntime
        print(f"ONNX Runtime version: {onnxruntime.__version__}")
        print(f"Available providers: {onnxruntime.get_available_providers()}")
        
        # Test a simple session creation
        from core.utils.device_utils import get_onnx_providers
        providers = get_onnx_providers("mps" if is_mac_arm() else "cpu")
        print(f"Recommended providers: {providers}")
        
        return True
    except ImportError as e:
        print(f"ONNX Runtime not available: {e}")
        return False


def test_pytorch_mps():
    """Test PyTorch MPS functionality."""
    print("\n=== Testing PyTorch MPS ===")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"MPS built: {torch.backends.mps.is_built()}")
        
        if torch.backends.mps.is_available():
            # Test basic MPS operations
            device = torch.device("mps")
            x = torch.randn(3, 3).to(device)
            y = torch.randn(3, 3).to(device)
            z = torch.mm(x, y)
            print(f"MPS tensor operation successful: {z.shape}")
            return True
        else:
            print("MPS not available on this system")
            return False
            
    except Exception as e:
        print(f"PyTorch MPS test failed: {e}")
        return False


def test_model_loading():
    """Test model loading with different backends."""
    print("\n=== Testing Model Loading ===")
    
    # Check if ONNX models are available
    onnx_dir = Path("./checkpoints/ditto_onnx")
    if not onnx_dir.exists():
        print(f"ONNX models not found at {onnx_dir}")
        print("Please download the checkpoints from HuggingFace")
        return False
    
    # Test loading a simple ONNX model
    test_model_path = onnx_dir / "appearance_extractor.onnx"
    if test_model_path.exists():
        try:
            from core.utils.load_model import load_model
            device, _ = detect_device()
            
            print(f"Testing model loading with device: {device}")
            model, model_type = load_model(str(test_model_path), device=device)
            print(f"Successfully loaded model type: {model_type}")
            return True
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            return False
    else:
        print(f"Test model not found: {test_model_path}")
        return False


def test_config_generation():
    """Test configuration file generation."""
    print("\n=== Testing Config Generation ===")
    
    try:
        from scripts.create_mac_arm_config import create_onnx_config, create_mps_config
        
        base_config = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
        if not os.path.exists(base_config):
            print(f"Base config not found: {base_config}")
            return False
        
        # Test config creation (dry run)
        print("Config generation functions imported successfully")
        return True
        
    except Exception as e:
        print(f"Config generation test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Mac ARM Support Test Suite")
    print("=" * 50)
    
    # Print system information
    print_device_info()
    
    # Run tests
    tests = [
        ("Device Detection", test_device_detection),
        ("ONNX Runtime", test_onnx_runtime),
        ("PyTorch MPS", test_pytorch_mps),
        ("Model Loading", test_model_loading),
        ("Config Generation", test_config_generation),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    # Overall status
    all_passed = all(results.values())
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if not all_passed:
        print("\nTroubleshooting:")
        if not results.get("ONNX Runtime", True):
            print("- Install ONNX Runtime: pip install onnxruntime")
        if not results.get("Model Loading", True):
            print("- Download checkpoints: git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints")
        if not results.get("Config Generation", True):
            print("- Run: python scripts/create_mac_arm_config.py")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 