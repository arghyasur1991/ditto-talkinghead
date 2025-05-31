import platform
import torch
import os


def detect_device():
    """
    Detect the best available device for inference.
    Priority: CUDA > MPS > CPU
    Returns device string and recommended model format.
    """
    # Check if CUDA is available
    if torch.cuda.is_available():
        return "cuda", "tensorrt"
    
    # Check if MPS is available (Mac ARM)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps", "pytorch"
    
    # Fallback to CPU
    return "cpu", "onnx"


def get_onnx_providers(device="cpu"):
    """
    Get appropriate ONNX runtime providers based on device.
    """
    import onnxruntime
    available_providers = onnxruntime.get_available_providers()
    
    if device == "cuda" and "CUDAExecutionProvider" in available_providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif device == "mps" and "CoreMLExecutionProvider" in available_providers:
        # For Mac ARM, use CoreML when available, fallback to CPU
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    else:
        return ["CPUExecutionProvider"]


def is_mac_arm():
    """
    Check if running on Mac ARM (Apple Silicon).
    """
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def get_recommended_config(device=None):
    """
    Get recommended configuration based on device.
    """
    if device is None:
        device, model_format = detect_device()
    else:
        if device == "cuda":
            model_format = "tensorrt"
        elif device == "mps":
            model_format = "pytorch"
        else:
            model_format = "onnx"
    
    config = {
        "device": device,
        "model_format": model_format,
        "onnx_providers": get_onnx_providers(device),
        "use_fp16": device in ["cuda", "mps"],  # Enable FP16 for GPU/MPS
        "batch_size": 1,  # Conservative batch size for compatibility
    }
    
    # Mac ARM specific optimizations
    if is_mac_arm():
        config.update({
            "torch_compile": False,  # Disable torch.compile for compatibility
            "autocast_enabled": device == "mps",  # Enable autocast for MPS
        })
    
    return config


def print_device_info():
    """
    Print detailed device information for debugging.
    """
    print("=== Device Information ===")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
    
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    try:
        import onnxruntime
        print(f"ONNX Runtime providers: {onnxruntime.get_available_providers()}")
    except ImportError:
        print("ONNX Runtime not available")
    
    device, model_format = detect_device()
    print(f"Recommended device: {device}")
    print(f"Recommended model format: {model_format}")
    print("=" * 30) 