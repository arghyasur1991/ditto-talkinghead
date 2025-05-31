#!/usr/bin/env python3
"""
Script to create Mac ARM compatible configuration files.
This script generates config files for both ONNX runtime and MPS (PyTorch) backends.
"""

import os
import pickle
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.utils.device_utils import detect_device, get_recommended_config, print_device_info


def create_cpu_safe_config(base_config_path, output_path, data_root="./checkpoints/ditto_onnx"):
    """Create CPU-only safe configuration that avoids device conflicts."""
    
    # Load the base TensorRT config
    with open(base_config_path, 'rb') as f:
        base_cfg = pickle.load(f)
    
    # Create new config based on the base
    new_cfg = base_cfg.copy()
    
    # Update all model paths to use ONNX versions (working ones)
    model_mappings = {
        'appearance_extractor_fp16.engine': 'appearance_extractor.onnx',
        'decoder_fp16.engine': 'decoder.onnx',
        'hubert_fp32.engine': 'hubert.onnx',
        'insightface_det_fp16.engine': 'insightface_det.onnx',
        'landmark106_fp16.engine': 'landmark106.onnx',
        'landmark203_fp16.engine': 'landmark203.onnx',
        'blaze_face_fp16.engine': 'blaze_face.onnx',
        'face_mesh_fp16.engine': 'face_mesh.onnx',
        'motion_extractor_fp32.engine': 'motion_extractor.onnx',
        'stitch_network_fp16.engine': 'stitch_network.onnx',
        'warp_network_fp16.engine': 'warp_network_ori.onnx',  # Use working version
        'lmdm_v0.4_hubert_fp32.engine': 'lmdm_v0.4_hubert.onnx',
        'wavlm_fp32.engine': 'wavlm.onnx'
    }
    
    # Update base_cfg model paths - force CPU for everything
    for component_name, component_cfg in new_cfg['base_cfg'].items():
        if 'model_path' in component_cfg:
            old_path = component_cfg['model_path']
            if old_path in model_mappings:
                component_cfg['model_path'] = model_mappings[old_path]
                print(f"Updated {component_name}: {old_path} -> {model_mappings[old_path]}")
        
        # Handle landmark478_cfg which has multiple model paths
        if component_name == 'landmark478_cfg':
            if 'blaze_face_model_path' in component_cfg:
                old_path = component_cfg['blaze_face_model_path']
                if old_path in model_mappings:
                    component_cfg['blaze_face_model_path'] = model_mappings[old_path]
                    print(f"Updated blaze_face: {old_path} -> {model_mappings[old_path]}")
            
            if 'face_mesh_model_path' in component_cfg:
                old_path = component_cfg['face_mesh_model_path']
                if old_path in model_mappings:
                    component_cfg['face_mesh_model_path'] = model_mappings[old_path]
                    print(f"Updated face_mesh: {old_path} -> {model_mappings[old_path]}")
        
        # Force CPU for all components to avoid device conflicts
        if 'device' in component_cfg:
            component_cfg['device'] = 'cpu'
    
    # Update audio2motion_cfg - force CPU
    if 'model_path' in new_cfg['audio2motion_cfg']:
        old_path = new_cfg['audio2motion_cfg']['model_path']
        if old_path in model_mappings:
            new_cfg['audio2motion_cfg']['model_path'] = model_mappings[old_path]
            print(f"Updated audio2motion: {old_path} -> {model_mappings[old_path]}")
    
    new_cfg['audio2motion_cfg']['device'] = 'cpu'
    
    # Save the new config
    with open(output_path, 'wb') as f:
        pickle.dump(new_cfg, f)
    
    print(f"Created CPU-safe config: {output_path}")


def create_onnx_config(base_config_path, output_path, data_root="./checkpoints/ditto_onnx"):
    """Create ONNX-based configuration for Mac ARM."""
    
    # Load the base TensorRT config
    with open(base_config_path, 'rb') as f:
        base_cfg = pickle.load(f)
    
    # Create new config based on the base
    new_cfg = base_cfg.copy()
    
    # Update all model paths to use ONNX versions
    model_mappings = {
        'appearance_extractor_fp16.engine': 'appearance_extractor.onnx',
        'decoder_fp16.engine': 'decoder.onnx',
        'hubert_fp32.engine': 'hubert.onnx',
        'insightface_det_fp16.engine': 'insightface_det.onnx',
        'landmark106_fp16.engine': 'landmark106.onnx',
        'landmark203_fp16.engine': 'landmark203.onnx',
        'blaze_face_fp16.engine': 'blaze_face.onnx',
        'face_mesh_fp16.engine': 'face_mesh.onnx',
        'motion_extractor_fp32.engine': 'motion_extractor.onnx',
        'stitch_network_fp16.engine': 'stitch_network.onnx',
        'warp_network_fp16.engine': 'warp_network_ori.onnx',  # Use alternative version
        'lmdm_v0.4_hubert_fp32.engine': 'lmdm_v0.4_hubert.onnx',
        'wavlm_fp32.engine': 'wavlm.onnx'  # If it exists
    }
    
    # Update base_cfg model paths
    for component_name, component_cfg in new_cfg['base_cfg'].items():
        if 'model_path' in component_cfg:
            old_path = component_cfg['model_path']
            if old_path in model_mappings:
                component_cfg['model_path'] = model_mappings[old_path]
                print(f"Updated {component_name}: {old_path} -> {model_mappings[old_path]}")
        
        # Handle landmark478_cfg which has multiple model paths
        if component_name == 'landmark478_cfg':
            if 'blaze_face_model_path' in component_cfg:
                old_path = component_cfg['blaze_face_model_path']
                if old_path in model_mappings:
                    component_cfg['blaze_face_model_path'] = model_mappings[old_path]
                    print(f"Updated blaze_face: {old_path} -> {model_mappings[old_path]}")
            
            if 'face_mesh_model_path' in component_cfg:
                old_path = component_cfg['face_mesh_model_path']
                if old_path in model_mappings:
                    component_cfg['face_mesh_model_path'] = model_mappings[old_path]
                    print(f"Updated face_mesh: {old_path} -> {model_mappings[old_path]}")
        
        # Update device to cpu for ONNX runtime
        if 'device' in component_cfg:
            component_cfg['device'] = 'cpu'
    
    # Update audio2motion_cfg
    if 'model_path' in new_cfg['audio2motion_cfg']:
        old_path = new_cfg['audio2motion_cfg']['model_path']
        if old_path in model_mappings:
            new_cfg['audio2motion_cfg']['model_path'] = model_mappings[old_path]
            print(f"Updated audio2motion: {old_path} -> {model_mappings[old_path]}")
    
    new_cfg['audio2motion_cfg']['device'] = 'cpu'
    
    # Save the new config
    with open(output_path, 'wb') as f:
        pickle.dump(new_cfg, f)
    
    print(f"Created ONNX config: {output_path}")


def create_mps_config(base_config_path, output_path, data_root="./checkpoints/ditto_onnx"):
    """Create MPS-based configuration for Mac ARM using PyTorch models."""
    
    # For MPS, we'll use ONNX models but with MPS device for PyTorch-compatible ones
    # This is a hybrid approach since not all models may have PyTorch versions
    
    # Load the base TensorRT config
    with open(base_config_path, 'rb') as f:
        base_cfg = pickle.load(f)
    
    # Create new config based on the base
    new_cfg = base_cfg.copy()
    
    # Models that can use MPS (PyTorch backend)
    mps_compatible_models = {
        'appearance_extractor_cfg',
        'decoder_cfg', 
        'motion_extractor_cfg',
        'stitch_network_cfg',
        'warp_network_cfg'
    }
    
    # Models that should use ONNX with CoreML provider
    onnx_models = {
        'insightface_det_cfg',
        'landmark106_cfg', 
        'landmark203_cfg',
        'landmark478_cfg',
        'hubert_cfg',
        'wavlm_cfg'
    }
    
    # Update model paths and devices
    model_mappings = {
        'appearance_extractor_fp16.engine': 'appearance_extractor.onnx',
        'decoder_fp16.engine': 'decoder.onnx',
        'hubert_fp32.engine': 'hubert.onnx',
        'insightface_det_fp16.engine': 'insightface_det.onnx',
        'landmark106_fp16.engine': 'landmark106.onnx',
        'landmark203_fp16.engine': 'landmark203.onnx',
        'blaze_face_fp16.engine': 'blaze_face.onnx',
        'face_mesh_fp16.engine': 'face_mesh.onnx',
        'motion_extractor_fp32.engine': 'motion_extractor.onnx',
        'stitch_network_fp16.engine': 'stitch_network.onnx',
        'warp_network_fp16.engine': 'warp_network_ori.onnx',  # Use alternative version
        'lmdm_v0.4_hubert_fp32.engine': 'lmdm_v0.4_hubert.onnx',
        'wavlm_fp32.engine': 'wavlm.onnx'
    }
    
    # Update base_cfg model paths and devices
    for component_name, component_cfg in new_cfg['base_cfg'].items():
        if 'model_path' in component_cfg:
            old_path = component_cfg['model_path']
            if old_path in model_mappings:
                component_cfg['model_path'] = model_mappings[old_path]
                print(f"Updated {component_name}: {old_path} -> {model_mappings[old_path]}")
        
        # Handle landmark478_cfg which has multiple model paths
        if component_name == 'landmark478_cfg':
            if 'blaze_face_model_path' in component_cfg:
                old_path = component_cfg['blaze_face_model_path']
                if old_path in model_mappings:
                    component_cfg['blaze_face_model_path'] = model_mappings[old_path]
            
            if 'face_mesh_model_path' in component_cfg:
                old_path = component_cfg['face_mesh_model_path']
                if old_path in model_mappings:
                    component_cfg['face_mesh_model_path'] = model_mappings[old_path]
        
        # Set device based on model compatibility
        if 'device' in component_cfg:
            if component_name in mps_compatible_models:
                component_cfg['device'] = 'mps'
                print(f"Set {component_name} to use MPS")
            else:
                component_cfg['device'] = 'cpu'  # Use CPU with ONNX CoreML provider
                print(f"Set {component_name} to use CPU/ONNX")
    
    # Update audio2motion_cfg - use ONNX for now since it's complex
    if 'model_path' in new_cfg['audio2motion_cfg']:
        old_path = new_cfg['audio2motion_cfg']['model_path']
        if old_path in model_mappings:
            new_cfg['audio2motion_cfg']['model_path'] = model_mappings[old_path]
            print(f"Updated audio2motion: {old_path} -> {model_mappings[old_path]}")
    
    new_cfg['audio2motion_cfg']['device'] = 'cpu'  # Use ONNX for LMDM for now
    
    # Save the new config
    with open(output_path, 'wb') as f:
        pickle.dump(new_cfg, f)
    
    print(f"Created MPS hybrid config: {output_path}")


def main():
    print("Creating Mac ARM compatible configurations...")
    print_device_info()
    
    # Paths
    base_config_path = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
    base_online_config_path = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl"
    
    output_dir = Path("./checkpoints/ditto_cfg")
    output_dir.mkdir(exist_ok=True)
    
    # Check if base config exists
    if not os.path.exists(base_config_path):
        print(f"Error: Base config not found at {base_config_path}")
        print("Please ensure you have downloaded the checkpoints from HuggingFace")
        return
    
    # Create CPU-safe config first (most compatible)
    cpu_safe_config_path = output_dir / "v0.4_hubert_cfg_cpu_safe.pkl"
    print("\n=== Creating CPU-Safe Config ===")
    create_cpu_safe_config(base_config_path, cpu_safe_config_path)
    
    # Create ONNX configs
    onnx_config_path = output_dir / "v0.4_hubert_cfg_onnx.pkl"
    onnx_online_config_path = output_dir / "v0.4_hubert_cfg_onnx_online.pkl"
    
    print("\n=== Creating ONNX Runtime Configs ===")
    create_onnx_config(base_config_path, onnx_config_path)
    
    if os.path.exists(base_online_config_path):
        create_onnx_config(base_online_config_path, onnx_online_config_path)
    
    # Create MPS configs
    mps_config_path = output_dir / "v0.4_hubert_cfg_mps.pkl"
    mps_online_config_path = output_dir / "v0.4_hubert_cfg_mps_online.pkl"
    
    print("\n=== Creating MPS Hybrid Configs ===")
    create_mps_config(base_config_path, mps_config_path)
    
    if os.path.exists(base_online_config_path):
        create_mps_config(base_online_config_path, mps_online_config_path)
    
    print("\n=== Configuration Creation Complete ===")
    print(f"CPU-safe config: {cpu_safe_config_path}")
    print(f"ONNX configs: {onnx_config_path}, {onnx_online_config_path}")
    print(f"MPS configs: {mps_config_path}, {mps_online_config_path}")
    print("\nRecommended usage for Mac ARM:")
    print(f"  --cfg_pkl {cpu_safe_config_path} --data_root ./checkpoints/ditto_onnx")
    print(f"  --cfg_pkl {onnx_config_path} --data_root ./checkpoints/ditto_onnx")
    print(f"  --cfg_pkl {mps_config_path} --data_root ./checkpoints/ditto_onnx")
    
    print("\nNote: CPU-safe config is the most stable option.")
    print("If you encounter issues, try the CPU-safe config first.")


if __name__ == "__main__":
    main() 