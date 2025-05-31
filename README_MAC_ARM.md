# Mac ARM (Apple Silicon) Support for Ditto TalkingHead

This document provides comprehensive instructions for running Ditto TalkingHead on Mac ARM (Apple Silicon) devices like M1, M2, M3, and M4 Macs.

## 🚀 Quick Start

### 1. Test Your System
```bash
python test_mac_arm.py
```

### 2. Generate Mac ARM Configs
```bash
python scripts/create_mac_arm_config.py
```

### 3. Run Inference
```bash
python inference.py \
    --audio_path "./example/audio.wav" \
    --source_path "./example/image.png" \
    --output_path "./tmp/result.mp4"
```

The system will automatically detect your hardware and select the best configuration!

## 🔧 Supported Backends

### 1. **MPS (Metal Performance Shaders)** - Recommended
- **Best performance** on Apple Silicon
- Uses GPU acceleration via Metal
- Hybrid approach: MPS for compatible models, ONNX+CoreML for others
- Automatic memory management

### 2. **ONNX Runtime with CoreML**
- Good performance with Apple's CoreML acceleration
- CPU fallback for maximum compatibility
- Lower memory usage
- Works on all Mac systems

### 3. **CPU Only**
- Fallback option for maximum compatibility
- Slower but works everywhere
- No special hardware requirements

## 📋 System Requirements

### Hardware
- Mac with Apple Silicon (M1, M2, M3, M4)
- 8GB+ RAM recommended (16GB+ for best performance)
- 10GB+ free disk space for models

### Software
- macOS 12.0+ (Monterey or later)
- Python 3.10+
- PyTorch 2.0+ with MPS support

## 🛠️ Installation

### 1. Clone and Setup Environment
```bash
git clone https://github.com/antgroup/ditto-talkinghead
cd ditto-talkinghead

# Create conda environment
conda env create -f environment.yaml
conda activate ditto

# Or install with pip (after installing PyTorch)
pip install onnxruntime librosa tqdm filetype imageio opencv-python-headless scikit-image
```

### 2. Download Models
```bash
# Download all checkpoints
git lfs install
git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints

# Or download specific models for Mac ARM
cd checkpoints
git sparse-checkout init --cone
git sparse-checkout set ditto_onnx ditto_cfg
git pull origin main
```

### 3. Generate Mac ARM Configurations
```bash
python scripts/create_mac_arm_config.py
```

This creates optimized configurations:
- `v0.4_hubert_cfg_onnx.pkl` - Pure ONNX runtime
- `v0.4_hubert_cfg_mps.pkl` - MPS hybrid (recommended)

## 🎯 Usage Examples

### Basic Usage (Auto-detection)
```bash
python inference.py \
    --audio_path "./example/audio.wav" \
    --source_path "./example/image.png" \
    --output_path "./output.mp4"
```

### Force Specific Backend
```bash
# Force MPS (if available)
python inference.py --device mps \
    --audio_path "./example/audio.wav" \
    --source_path "./example/image.png" \
    --output_path "./output.mp4"

# Force ONNX Runtime
python inference.py --device cpu \
    --cfg_pkl "./checkpoints/ditto_cfg/v0.4_hubert_cfg_onnx.pkl" \
    --data_root "./checkpoints/ditto_onnx" \
    --audio_path "./example/audio.wav" \
    --source_path "./example/image.png" \
    --output_path "./output.mp4"
```

### Check Device Information
```bash
python inference.py --print_device_info
```

## ⚡ Performance Optimization

### MPS Optimization Tips
1. **Memory Management**: The system automatically manages MPS memory
2. **Batch Size**: Uses conservative batch size (1) for stability
3. **Mixed Precision**: Automatically disabled for MPS compatibility
4. **Model Compilation**: Disabled for better compatibility

### ONNX Runtime Optimization
1. **CoreML Provider**: Automatically used when available
2. **Provider Options**: Optimized for Mac ARM architecture
3. **Memory Limits**: Conservative settings for stability

### General Tips
1. **Close Other Apps**: Free up memory for better performance
2. **Monitor Temperature**: Use Activity Monitor to check system load
3. **Storage**: Ensure sufficient free disk space (10GB+)

## 🐛 Troubleshooting

### Common Issues

#### 1. "MPS not available"
```bash
# Check PyTorch MPS support
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# Update PyTorch if needed
conda update pytorch torchvision torchaudio -c pytorch
```

#### 2. "ONNX Runtime providers not found"
```bash
# Install/update ONNX Runtime
pip install --upgrade onnxruntime

# Check available providers
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

#### 3. "Config file not found"
```bash
# Generate Mac ARM configs
python scripts/create_mac_arm_config.py

# Check if base configs exist
ls -la checkpoints/ditto_cfg/
```

#### 4. "Model loading failed"
```bash
# Test model loading
python test_mac_arm.py

# Check model files
ls -la checkpoints/ditto_onnx/
```

#### 5. Memory Issues
- Reduce batch size in config
- Close other applications
- Use ONNX runtime instead of MPS
- Monitor memory usage with Activity Monitor

### Performance Issues
1. **Slow inference**: Try ONNX runtime with CoreML
2. **High memory usage**: Use CPU-only mode
3. **System overheating**: Reduce concurrent processes

## 📊 Performance Comparison

| Backend | Speed | Memory | Compatibility | Recommended For |
|---------|-------|--------|---------------|-----------------|
| MPS Hybrid | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | M1/M2/M3/M4 Macs |
| ONNX+CoreML | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | All Macs |
| CPU Only | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Compatibility |

## 🔍 Technical Details

### Device Detection Logic
1. Check CUDA availability (for compatibility)
2. Check MPS availability and build status
3. Fallback to CPU with ONNX runtime
4. Auto-select appropriate model format

### Model Loading Strategy
- **PyTorch models**: Use MPS when available
- **ONNX models**: Use CoreML provider on Mac ARM
- **Autocast handling**: Disabled for MPS, optimized for CUDA
- **Memory management**: Conservative settings for stability

### Configuration Generation
- Automatically maps TensorRT models to ONNX equivalents
- Sets appropriate devices for each model component
- Handles special cases (landmark478, audio2motion)
- Creates both pure ONNX and MPS hybrid configs

## 🤝 Contributing

If you encounter issues or have improvements for Mac ARM support:

1. Run the test suite: `python test_mac_arm.py`
2. Check the troubleshooting section
3. Open an issue with system information and error logs
4. Include output from `python inference.py --print_device_info`

## 📝 Changelog

### v1.0.0 - Mac ARM Support
- Added automatic device detection
- Implemented MPS backend support
- Created ONNX runtime optimization for Mac ARM
- Added CoreML provider integration
- Improved memory management for Apple Silicon
- Created comprehensive test suite
- Added automatic configuration generation

## 📚 Additional Resources

- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [ONNX Runtime Execution Providers](https://onnxruntime.ai/docs/execution-providers/)
- [Apple CoreML Documentation](https://developer.apple.com/documentation/coreml)
- [Mac ARM Performance Guide](https://developer.apple.com/documentation/apple-silicon)

---

**Note**: This Mac ARM support is designed to provide the best possible performance on Apple Silicon while maintaining compatibility with the original codebase. The hybrid approach ensures optimal performance by using the best backend for each model component. 