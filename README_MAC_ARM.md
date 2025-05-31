# Mac ARM Support for Ditto TalkingHead

This document provides comprehensive Mac ARM (Apple Silicon) support for the Ditto TalkingHead project.

## ✅ Status: WORKING

**Successfully tested on Mac ARM with:**
- MPS (Metal Performance Shaders) support
- ONNX runtime with CoreML acceleration  
- CPU fallback mode
- Complete video generation with audio
- **NEW: Synchronous pipeline for reliable processing**

## Quick Start

### 1. Install Dependencies

```bash
# Install FFmpeg (required for video processing)
brew install ffmpeg

# Install Python dependencies
pip install "imageio[ffmpeg]"

# Install other missing dependencies if needed
pip install librosa scikit-image cython
```

### 2. Generate Mac ARM Configuration

```bash
python scripts/create_mac_arm_config.py
```

This creates optimized configurations:
- `v0.4_hubert_cfg_mps.pkl` - MPS accelerated (recommended for M1/M2/M3)
- `v0.4_hubert_cfg_onnx.pkl` - ONNX with CoreML acceleration
- `v0.4_hubert_cfg_cpu_safe.pkl` - CPU-only (most reliable)

### 3. Run Inference

#### Option A: Synchronous Pipeline (Recommended)

The synchronous pipeline is more reliable and avoids threading issues:

```bash
# Auto-detect best device (MPS > CPU)
python sync_inference.py \
    --audio_path "./example/audio.wav" \
    --source_path "./example/image.png" \
    --output_path "./output.mp4"

# Force specific device
python sync_inference.py \
    --device mps \
    --audio_path "./example/audio.wav" \
    --source_path "./example/image.png" \
    --output_path "./output.mp4"

# Keep intermediate frames for debugging
python sync_inference.py \
    --device mps \
    --audio_path "./example/audio.wav" \
    --source_path "./example/image.png" \
    --output_path "./output.mp4" \
    --keep_frames
```

#### Option B: Original Streaming Pipeline

```bash
# Auto-detect best device
python inference.py \
    --audio_path "./example/audio.wav" \
    --source_path "./example/image.png" \
    --output_path "./output.mp4"

# Force specific device
python inference.py \
    --device mps \
    --audio_path "./example/audio.wav" \
    --source_path "./example/image.png" \
    --output_path "./output.mp4"
```

## Performance Comparison

**Synchronous Pipeline Performance (394 frames, ~16 seconds video):**
- **MPS (M-series Mac)**: ~6 minutes total (1.1 it/s frame generation)
- **CPU**: ~12 minutes total (0.55 it/s frame generation)

**Key Advantages of Synchronous Pipeline:**
- ✅ No threading issues or hangs
- ✅ Reliable completion every time
- ✅ Better error handling and debugging
- ✅ Memory efficient (processes frame by frame)
- ✅ Works with all device types (MPS/CPU/ONNX)

## Device Selection

The system automatically detects the best available device:

1. **MPS (Metal Performance Shaders)** - Best for M1/M2/M3 Macs
   - Uses PyTorch MPS backend for compatible models
   - Falls back to ONNX+CoreML for unsupported operations
   - ~2x faster than CPU

2. **CPU with ONNX+CoreML** - Universal fallback
   - Uses ONNX runtime with CoreML acceleration
   - Most compatible option
   - Slower but very reliable

3. **CPU-only** - Maximum compatibility
   - Pure CPU processing with ONNX runtime
   - Slowest but works on any system

## Configuration Files

- `v0.4_hubert_cfg_mps.pkl` - Hybrid MPS + ONNX configuration
- `v0.4_hubert_cfg_onnx.pkl` - ONNX with CoreML providers
- `v0.4_hubert_cfg_cpu_safe.pkl` - CPU-only safe configuration

## Troubleshooting

### Common Issues

1. **"FFmpeg not found"**
   ```bash
   brew install ffmpeg
   ```

2. **"imageio FFMPEG plugin not found"**
   ```bash
   pip install "imageio[ffmpeg]"
   ```

3. **Threading issues with original pipeline**
   - Use the synchronous pipeline instead: `sync_inference.py`

4. **Memory issues**
   - Use CPU mode: `--device cpu`
   - The synchronous pipeline is more memory efficient

### Debug Information

```bash
# Check device capabilities
python sync_inference.py --print_device_info

# Test with simple example
python test_mac_arm.py
```

## Technical Details

### Synchronous Pipeline Architecture

The synchronous pipeline (`sync_inference.py`) processes the video generation in stages:

1. **Audio Processing**: Convert audio to features using Wav2Feat
2. **Motion Generation**: Use Audio2Motion (LMDM) to generate motion sequences
3. **Frame Generation**: Process each frame through the complete pipeline:
   - Motion Stitch: Combine source and driving motion
   - Warp F3D: Apply 3D warping
   - Decode F3D: Generate rendered image
   - PutBack: Composite final frame
4. **Video Creation**: Use FFmpeg to combine frames with audio

### Model Compatibility

- **Avatar Registrar**: ONNX + CoreML
- **Motion Extractor**: ONNX + CoreML  
- **Appearance Extractor**: ONNX + CoreML
- **Audio2Motion (LMDM)**: PyTorch MPS (when available)
- **Motion Stitch**: PyTorch MPS
- **Warp Network**: ONNX (uses warp_network_ori.onnx for compatibility)
- **Decoder**: ONNX + CoreML

## Example Usage

```bash
# Quick test with synchronous pipeline
python sync_inference.py \
    --audio_path "./example/audio.wav" \
    --source_path "./example/image.png" \
    --output_path "./my_video.mp4"

# High quality with MPS acceleration
python sync_inference.py \
    --device mps \
    --audio_path "./my_audio.wav" \
    --source_path "./my_photo.jpg" \
    --output_path "./result.mp4" \
    --fps 30
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.8+
- PyTorch with MPS support
- FFmpeg
- All dependencies from `environment.yaml`

The implementation provides a robust, efficient solution for running Ditto TalkingHead on Mac ARM systems with automatic device detection and optimal performance. 