import librosa
import math
import os
import numpy as np
import random
import torch
import pickle

from stream_pipeline_offline import StreamSDK
from core.utils.device_utils import detect_device, get_recommended_config, print_device_info


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pkl(pkl):
    with open(pkl, "rb") as f:
        return pickle.load(f)


def auto_select_config(cfg_pkl_arg, data_root_arg, forced_device=None):
    """
    Automatically select the best configuration based on available hardware.
    """
    if forced_device:
        device = forced_device
        if device == "cuda":
            model_format = "tensorrt"
        elif device == "mps":
            model_format = "pytorch"
        else:
            model_format = "onnx"
    else:
        device, model_format = detect_device()
    
    # If user provided specific config, use it
    if cfg_pkl_arg and os.path.exists(cfg_pkl_arg):
        return cfg_pkl_arg, data_root_arg
    
    # Auto-select based on device
    config_dir = "./checkpoints/ditto_cfg"
    
    if device == "cuda":
        # Use TensorRT if available
        trt_config = os.path.join(config_dir, "v0.4_hubert_cfg_trt.pkl")
        if os.path.exists(trt_config):
            return trt_config, "./checkpoints/ditto_trt_Ampere_Plus"
        else:
            print("Warning: TensorRT config not found, falling back to ONNX")
            device = "cpu"
    
    if device == "mps":
        # Try MPS hybrid config first
        mps_config = os.path.join(config_dir, "v0.4_hubert_cfg_mps.pkl")
        if os.path.exists(mps_config):
            return mps_config, "./checkpoints/ditto_onnx"
        else:
            print("Warning: MPS config not found, falling back to ONNX")
            device = "cpu"
    
    if device == "cpu" or model_format == "onnx":
        # Try CPU-safe config first (most reliable)
        cpu_safe_config = os.path.join(config_dir, "v0.4_hubert_cfg_cpu_safe.pkl")
        if os.path.exists(cpu_safe_config):
            return cpu_safe_config, "./checkpoints/ditto_onnx"
        
        # Fallback to regular ONNX config
        onnx_config = os.path.join(config_dir, "v0.4_hubert_cfg_onnx.pkl")
        if os.path.exists(onnx_config):
            return onnx_config, "./checkpoints/ditto_onnx"
        else:
            print("Warning: ONNX config not found, using original TensorRT config")
            return os.path.join(config_dir, "v0.4_hubert_cfg_trt.pkl"), data_root_arg or "./checkpoints/ditto_trt_Ampere_Plus"
    
    # Fallback to original
    return cfg_pkl_arg or os.path.join(config_dir, "v0.4_hubert_cfg_trt.pkl"), data_root_arg or "./checkpoints/ditto_trt_Ampere_Plus"


def run(SDK: StreamSDK, audio_path: str, source_path: str, output_path: str, more_kwargs: str | dict = {}):
    print("DEBUG: Starting run function")

    if isinstance(more_kwargs, str):
        more_kwargs = load_pkl(more_kwargs)
    setup_kwargs = more_kwargs.get("setup_kwargs", {})
    run_kwargs = more_kwargs.get("run_kwargs", {})

    print("DEBUG: Calling SDK.setup")
    SDK.setup(source_path, output_path, **setup_kwargs)
    print("DEBUG: SDK.setup completed")

    print("DEBUG: Loading audio")
    audio, sr = librosa.core.load(audio_path, sr=16000)
    num_f = math.ceil(len(audio) / 16000 * 25)
    print(f"DEBUG: Audio loaded, {len(audio)} samples, {num_f} frames")

    fade_in = run_kwargs.get("fade_in", -1)
    fade_out = run_kwargs.get("fade_out", -1)
    ctrl_info = run_kwargs.get("ctrl_info", {})
    print("DEBUG: Calling SDK.setup_Nd")
    SDK.setup_Nd(N_d=num_f, fade_in=fade_in, fade_out=fade_out, ctrl_info=ctrl_info)
    print("DEBUG: SDK.setup_Nd completed")

    online_mode = SDK.online_mode
    print(f"DEBUG: Online mode: {online_mode}")
    
    if online_mode:
        print("DEBUG: Running in online mode")
        chunksize = run_kwargs.get("chunksize", (3, 5, 2))
        audio = np.concatenate([np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0)
        split_len = int(sum(chunksize) * 0.04 * 16000) + 80  # 6480
        for i in range(0, len(audio), chunksize[1] * 640):
            print(f"DEBUG: Processing chunk {i//chunksize[1]//640}")
            audio_chunk = audio[i:i + split_len]
            if len(audio_chunk) < split_len:
                audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
            SDK.run_chunk(audio_chunk, chunksize)
    else:
        print("DEBUG: Running in offline mode")
        print("DEBUG: Converting audio to features")
        aud_feat = SDK.wav2feat.wav2feat(audio)
        print(f"DEBUG: Audio features shape: {aud_feat.shape}")
        print("DEBUG: Putting audio features in queue")
        SDK.audio2motion_queue.put(aud_feat)
        print("DEBUG: Audio features queued")
    
    print("DEBUG: Calling SDK.close")
    SDK.close()
    print("DEBUG: SDK.close completed")

    print("DEBUG: Running ffmpeg")
    cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
    print(cmd)
    os.system(cmd)
    print("DEBUG: ffmpeg completed")

    print(output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None, help="path to model data_root (auto-detected if not provided)")
    parser.add_argument("--cfg_pkl", type=str, default=None, help="path to cfg_pkl (auto-detected if not provided)")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu"], help="force specific device (auto-detected if not provided)")
    parser.add_argument("--print_device_info", action="store_true", help="print device information and exit")

    parser.add_argument("--audio_path", type=str, help="path to input wav")
    parser.add_argument("--source_path", type=str, help="path to input image")
    parser.add_argument("--output_path", type=str, help="path to output mp4")
    args = parser.parse_args()

    # Print device info if requested
    if args.print_device_info:
        print_device_info()
        exit(0)

    # Auto-detect device and configuration
    if args.device:
        print(f"Using forced device: {args.device}")
        device = args.device
    else:
        device, model_format = detect_device()
        print(f"Auto-detected device: {device} (model format: {model_format})")

    # Auto-select configuration
    cfg_pkl, data_root = auto_select_config(args.cfg_pkl, args.data_root, args.device)
    print(f"Using config: {cfg_pkl}")
    print(f"Using data root: {data_root}")

    # Validate paths
    if not os.path.exists(cfg_pkl):
        print(f"Error: Config file not found: {cfg_pkl}")
        print("Please run: python scripts/create_mac_arm_config.py")
        exit(1)

    if not os.path.exists(data_root):
        print(f"Error: Data root not found: {data_root}")
        print("Please ensure you have downloaded the checkpoints from HuggingFace")
        exit(1)

    # Validate input arguments
    if not args.audio_path or not args.source_path or not args.output_path:
        print("Error: audio_path, source_path, and output_path are required")
        parser.print_help()
        exit(1)

    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print("DEBUG: Initializing SDK")
    # init sdk
    SDK = StreamSDK(cfg_pkl, data_root)
    print("DEBUG: SDK initialized")

    # input args
    audio_path = args.audio_path    # .wav
    source_path = args.source_path   # video|image
    output_path = args.output_path   # .mp4

    # run
    # seed_everything(1024)
    print("DEBUG: Starting inference")
    run(SDK, audio_path, source_path, output_path)
