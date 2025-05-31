from .device_utils import get_onnx_providers


def load_model(model_path: str, device: str = "cuda", **kwargs):
    if kwargs.get("force_ori_type", False):
        # for hubert, landmark, retinaface, mediapipe
        model = load_force_ori_type(model_path, device, **kwargs)
        return model, "ori"

    if model_path.endswith(".onnx"):
        # onnx
        import onnxruntime

        # Use device-aware provider selection
        providers = get_onnx_providers(device)
        
        # Set provider options for better performance
        provider_options = []
        for provider in providers:
            if provider == "CoreMLExecutionProvider":
                # CoreML specific options for Mac ARM - use minimal options
                provider_options.append({})
            elif provider == "CUDAExecutionProvider":
                # CUDA specific options
                provider_options.append({
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB limit
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                })
            else:
                provider_options.append({})

        try:
            model = onnxruntime.InferenceSession(
                model_path, 
                providers=list(zip(providers, provider_options))
            )
        except Exception as e:
            print(f"Warning: Failed to load with preferred providers, falling back to CPU: {e}")
            model = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        
        return model, "onnx"

    elif model_path.endswith(".engine") or model_path.endswith(".trt"):
        # tensorRT - only works on CUDA
        if device != "cuda":
            raise ValueError(f"TensorRT models require CUDA device, got {device}")
        
        from .tensorrt_utils import TRTWrapper
        model = TRTWrapper(model_path)
        return model, "tensorrt"

    elif model_path.endswith(".pt") or model_path.endswith(".pth"):
        # pytorch
        model = create_model(model_path, device, **kwargs)
        return model, "pytorch"

    else:
        raise ValueError(f"Unsupported model file type: {model_path}")


def create_model(
    model_path: str,
    device: str = "cuda",
    module_name="",
    package_name="..models.modules",
    **kwargs,
):
    import importlib
    import torch

    # module = getattr(importlib.import_module('..models.modules', __package__), module_name)
    module = getattr(importlib.import_module(package_name, __package__), module_name)
    # from <package_name> import <module_name>

    model = module(**kwargs)
    model.load_model(model_path)
    
    # Handle device placement - MPS compatible
    if device == "mps":
        # Ensure model is on MPS device
        model = model.to(device)
        # Enable MPS optimizations
        if hasattr(torch.backends.mps, 'set_per_process_memory_fraction'):
            torch.backends.mps.set_per_process_memory_fraction(0.8)
    elif device == "cuda":
        model = model.to(device)
    else:
        model = model.to("cpu")
    
    return model


def load_force_ori_type(
    model_path: str,
    device: str = "cuda",
    module_name="",
    package_name="..aux_models.modules",
    force_ori_type=False, 
    **kwargs,
):
    import importlib

    module = getattr(importlib.import_module(package_name, __package__), module_name)
    model = module(**kwargs)
    return model
