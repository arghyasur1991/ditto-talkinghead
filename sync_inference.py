#!/usr/bin/env python3
"""
Synchronous inference pipeline for Mac ARM.
This version generates all images first, then uses ffmpeg to stitch them together.
"""

import librosa
import math
import os
import numpy as np
import random
import torch
import pickle
import cv2
from tqdm import tqdm
from pathlib import Path

from core.utils.device_utils import detect_device, get_recommended_config, print_device_info


def _mirror_index(idx, size):
    """Mirror index for cycling through frames."""
    return idx % size


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


class SyncStreamSDK:
    """Synchronous version of StreamSDK that generates all frames before video creation."""
    
    def __init__(self, cfg_pkl, data_root):
        print("Initializing Synchronous SDK...")
        
        # Import and initialize components
        from core.atomic_components.avatar_registrar import AvatarRegistrar
        from core.atomic_components.condition_handler import ConditionHandler
        from core.atomic_components.audio2motion import Audio2Motion
        from core.atomic_components.motion_stitch import MotionStitch
        from core.atomic_components.warp_f3d import WarpF3D
        from core.atomic_components.decode_f3d import DecodeF3D
        from core.atomic_components.putback import PutBack
        from core.atomic_components.wav2feat import Wav2Feat
        from core.atomic_components.cfg import parse_cfg
        
        print("Parsing configuration...")
        
        # Parse configuration like the original StreamSDK
        [
            avatar_registrar_cfg,
            condition_handler_cfg,
            lmdm_cfg,
            stitch_network_cfg,
            warp_network_cfg,
            decoder_cfg,
            wav2feat_cfg,
            default_kwargs,
        ] = parse_cfg(cfg_pkl, data_root, {})
        
        self.default_kwargs = default_kwargs
        
        print("Loading models...")
        
        # Initialize components
        self.avatar_registrar = AvatarRegistrar(**avatar_registrar_cfg)
        self.condition_handler = ConditionHandler(**condition_handler_cfg)
        self.audio2motion = Audio2Motion(lmdm_cfg)
        self.motion_stitch = MotionStitch(stitch_network_cfg)
        self.warp_f3d = WarpF3D(warp_network_cfg)
        self.decode_f3d = DecodeF3D(decoder_cfg)
        self.putback = PutBack()
        self.wav2feat = Wav2Feat(**wav2feat_cfg)
        
        print("Models loaded successfully!")
        
    def setup(self, source_path, output_dir, **kwargs):
        """Setup the pipeline for synchronous processing."""
        print("Setting up synchronous pipeline...")
        
        # Merge kwargs
        kwargs = self._merge_kwargs(self.default_kwargs, kwargs)
        self.kwargs = kwargs
        
        # Extract configuration parameters like the original StreamSDK
        self.max_size = kwargs.get("max_size", 1920)
        self.template_n_frames = kwargs.get("template_n_frames", -1)
        self.crop_scale = kwargs.get("crop_scale", 2.3)
        self.crop_vx_ratio = kwargs.get("crop_vx_ratio", 0)
        self.crop_vy_ratio = kwargs.get("crop_vy_ratio", -0.125)
        self.crop_flag_do_rot = kwargs.get("crop_flag_do_rot", True)
        self.smo_k_s = kwargs.get('smo_k_s', 13)
        self.emo = kwargs.get("emo", 4)
        self.eye_f0_mode = kwargs.get("eye_f0_mode", False)
        self.ch_info = kwargs.get("ch_info", None)
        self.overlap_v2 = kwargs.get("overlap_v2", 10)
        self.fix_kp_cond = kwargs.get("fix_kp_cond", 0)
        self.fix_kp_cond_dim = kwargs.get("fix_kp_cond_dim", None)
        self.sampling_timesteps = kwargs.get("sampling_timesteps", 50)
        self.online_mode = kwargs.get("online_mode", False)
        self.v_min_max_for_clip = kwargs.get('v_min_max_for_clip', None)
        self.smo_k_d = kwargs.get("smo_k_d", 3)
        self.N_d = kwargs.get("N_d", -1)
        self.use_d_keys = kwargs.get("use_d_keys", None)
        self.relative_d = kwargs.get("relative_d", True)
        self.drive_eye = kwargs.get("drive_eye", None)
        self.delta_eye_arr = kwargs.get("delta_eye_arr", None)
        self.delta_eye_open_n = kwargs.get("delta_eye_open_n", 0)
        self.fade_type = kwargs.get("fade_type", "")
        self.fade_out_keys = kwargs.get("fade_out_keys", ("exp",))
        self.flag_stitching = kwargs.get("flag_stitching", True)
        self.ctrl_info = kwargs.get("ctrl_info", dict())
        self.overall_ctrl_info = kwargs.get("overall_ctrl_info", dict())
        
        # Setup avatar registration
        print("Registering avatar...")
        crop_kwargs = {
            "crop_scale": self.crop_scale,
            "crop_vx_ratio": self.crop_vx_ratio,
            "crop_vy_ratio": self.crop_vy_ratio,
            "crop_flag_do_rot": self.crop_flag_do_rot,
        }
        n_frames = self.template_n_frames if self.template_n_frames > 0 else self.N_d
        self.source_info = self.avatar_registrar(
            source_path, 
            max_dim=self.max_size, 
            n_frames=n_frames, 
            **crop_kwargs,
        )
        print(f"Avatar registered with keys: {list(self.source_info.keys())}")
        if 'frame_height' in self.source_info and 'frame_width' in self.source_info:
            print(f"Avatar registered: {self.source_info['frame_height']}x{self.source_info['frame_width']}")
        else:
            print("Avatar registered successfully")
        
        # Smooth x_s_info_lst if needed
        if len(self.source_info["x_s_info_lst"]) > 1 and self.smo_k_s > 1:
            from core.atomic_components.avatar_registrar import smooth_x_s_info_lst
            self.source_info["x_s_info_lst"] = smooth_x_s_info_lst(self.source_info["x_s_info_lst"], smo_k=self.smo_k_s)
        
        self.source_info_frames = len(self.source_info["x_s_info_lst"])
        
        # Setup condition handler
        print("Setting up condition handler...")
        self.condition_handler.setup(self.source_info, self.emo, eye_f0_mode=self.eye_f0_mode, ch_info=self.ch_info)
        
        # Setup Audio2Motion
        print("Setting up Audio2Motion...")
        x_s_info_0 = self.condition_handler.x_s_info_0
        self.audio2motion.setup(
            x_s_info_0, 
            overlap_v2=self.overlap_v2,
            fix_kp_cond=self.fix_kp_cond,
            fix_kp_cond_dim=self.fix_kp_cond_dim,
            sampling_timesteps=self.sampling_timesteps,
            online_mode=self.online_mode,
            v_min_max_for_clip=self.v_min_max_for_clip,
            smo_k_d=self.smo_k_d,
        )
        
        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear any existing frames
        for frame_file in self.output_dir.glob("frame_*.png"):
            frame_file.unlink()
            
        print(f"Output directory: {self.output_dir}")
        
    def _merge_kwargs(self, default_kwargs, user_kwargs):
        """Merge user kwargs with defaults."""
        merged = default_kwargs.copy()
        merged.update(user_kwargs)
        return merged
    
    def setup_motion_stitch(self):
        """Setup motion stitch after N_d is known."""
        print("Setting up Motion Stitch...")
        is_image_flag = self.source_info["is_image_flag"]
        x_s_info = self.source_info['x_s_info_lst'][0]
        self.motion_stitch.setup(
            N_d=self.N_d,
            use_d_keys=self.use_d_keys,
            relative_d=self.relative_d,
            drive_eye=self.drive_eye,
            delta_eye_arr=self.delta_eye_arr,
            delta_eye_open_n=self.delta_eye_open_n,
            fade_out_keys=self.fade_out_keys,
            fade_type=self.fade_type,
            flag_stitching=self.flag_stitching,
            is_image_flag=is_image_flag,
            x_s_info=x_s_info,
            d0=None,
            ch_info=self.ch_info,
            overall_ctrl_info=self.overall_ctrl_info,
        )
    
    def setup_Nd(self, N_d, fade_in=-1, fade_out=-1, ctrl_info=None):
        """Setup the number of frames and control info."""
        self.N_d = N_d
        self.fade_in = fade_in
        self.fade_out = fade_out
        
        # Merge with existing ctrl_info
        if ctrl_info:
            self.ctrl_info.update(ctrl_info)
        
        # For fade in/out alpha
        if fade_in > 0:
            for i in range(fade_in):
                alpha = i / fade_in
                item = self.ctrl_info.get(i, {})
                item["fade_alpha"] = alpha
                self.ctrl_info[i] = item
        if fade_out > 0:
            ss = N_d - fade_out - 1
            ee = N_d - 1
            for i in range(ss, N_d):
                alpha = max((ee - i) / (ee - ss), 0)
                item = self.ctrl_info.get(i, {})
                item["fade_alpha"] = alpha
                self.ctrl_info[i] = item
                
        print(f"Frame count: {N_d}")
        
        # Now setup motion stitch with the frame count
        self.setup_motion_stitch()
        
        # For eye open at video end (after motion_stitch is set up)
        self.motion_stitch.set_Nd(N_d)
    
    def process_audio_to_motion(self, audio_features):
        """Convert audio features to motion using Audio2Motion."""
        print("Processing audio to motion...")
        
        # Follow the same pattern as _audio2motion_offline
        aud_cond_all = self.condition_handler(audio_features, 0)
        seq_frames = self.audio2motion.seq_frames
        valid_clip_len = self.audio2motion.valid_clip_len
        num_frames = len(aud_cond_all)
        idx = 0
        res_kp_seq = None
        
        print(f"Processing {num_frames} frames with seq_frames={seq_frames}, valid_clip_len={valid_clip_len}")
        pbar = tqdm(desc="dit")
        
        while idx < num_frames:
            pbar.update()
            aud_cond = aud_cond_all[idx:idx + seq_frames][None]
            if aud_cond.shape[1] < seq_frames:
                pad = np.stack([aud_cond[:, -1]] * (seq_frames - aud_cond.shape[1]), 1)
                aud_cond = np.concatenate([aud_cond, pad], 1)
            res_kp_seq = self.audio2motion(aud_cond, res_kp_seq)
            idx += valid_clip_len

        pbar.close()
        res_kp_seq = res_kp_seq[:, :num_frames]
        res_kp_seq = self.audio2motion._smo(res_kp_seq, 0, res_kp_seq.shape[1])

        x_d_info_list = self.audio2motion.cvt_fmt(res_kp_seq)
        
        print(f"Motion processing complete: {len(x_d_info_list)} motion frames")
        return x_d_info_list
    
    def stitch_motion(self, x_d_info_list):
        """The motion stitching is now done per frame, so we'll handle it in generate_frames."""
        print("Motion data ready for per-frame processing...")
        return x_d_info_list
    
    def generate_frames(self, x_d_info_list):
        """Generate all video frames synchronously."""
        print("Generating frames...")
        
        frame_count = len(x_d_info_list)
        
        for gen_frame_idx in tqdm(range(frame_count), desc="Generating frames"):
            # Get the motion data for this frame
            x_d_info = x_d_info_list[gen_frame_idx]
            
            # Follow the complete pipeline: motion -> stitch -> warp -> decode -> putback
            with torch.no_grad():
                # Step 1: Motion stitch - get x_s and x_d from motion
                source_frame_idx = _mirror_index(gen_frame_idx, self.source_info_frames)  # Mirror index
                x_s_info = self.source_info["x_s_info_lst"][source_frame_idx]
                
                # Get control info for this frame
                ctrl_kwargs = self._get_ctrl_info(gen_frame_idx)
                
                # Motion stitch to get x_s and x_d
                x_s, x_d = self.motion_stitch(x_s_info, x_d_info, **ctrl_kwargs)
                
                # Step 2: Warp using x_s and x_d
                f_s = self.source_info["f_s_lst"][source_frame_idx]
                f_3d = self.warp_f3d(f_s, x_s, x_d)
                
                # Step 3: Decode the warped features  
                render_img = self.decode_f3d(f_3d)
                
                # Step 4: Put back to final image
                frame_rgb = self.source_info["img_rgb_lst"][source_frame_idx]
                M_c2o = self.source_info["M_c2o_lst"][source_frame_idx]
                res_frame_rgb = self.putback(frame_rgb, render_img, M_c2o)
            
            # Convert to image for saving
            if isinstance(res_frame_rgb, torch.Tensor):
                frame_image = res_frame_rgb.cpu().numpy()
            else:
                frame_image = res_frame_rgb
            
            # Ensure proper format (H, W, C) and range [0, 255]
            if frame_image.max() <= 1.0:
                frame_image = (frame_image * 255).astype(np.uint8)
            else:
                frame_image = frame_image.astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            frame_image_bgr = cv2.cvtColor(frame_image, cv2.COLOR_RGB2BGR)
            
            # Save frame
            frame_path = self.output_dir / f"frame_{gen_frame_idx:06d}.png"
            cv2.imwrite(str(frame_path), frame_image_bgr)
        
        print(f"Generated {frame_count} frames in {self.output_dir}")
        return frame_count
    
    def _get_ctrl_info(self, fid):
        """Get control info for a frame."""
        try:
            if isinstance(self.ctrl_info, dict):
                return self.ctrl_info.get(fid, {})
            elif isinstance(self.ctrl_info, list):
                return self.ctrl_info[fid]
            else:
                return {}
        except Exception:
            return {}
    
    def create_video_with_ffmpeg(self, frame_count, audio_path, output_path, fps=25):
        """Use ffmpeg to create video from frames and audio."""
        print("Creating video with ffmpeg...")
        
        # FFmpeg command to create video from images and add audio
        cmd = [
            'ffmpeg', '-y',  # Overwrite output file
            '-framerate', str(fps),  # Input framerate
            '-i', str(self.output_dir / 'frame_%06d.png'),  # Input image pattern
            '-i', audio_path,  # Input audio
            '-c:v', 'libx264',  # Video codec
            '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
            '-c:a', 'aac',  # Audio codec
            '-shortest',  # Stop encoding when shortest input ends
            output_path
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Video created successfully: {output_path}")
            return True
        else:
            print(f"FFmpeg error: {result.stderr}")
            return False


def run_sync(cfg_pkl, data_root, audio_path, source_path, output_path, frames_dir=None, fps=25, **kwargs):
    """Run the synchronous inference pipeline."""
    
    # Create frames directory
    if frames_dir is None:
        frames_dir = Path(output_path).parent / "frames"
    else:
        frames_dir = Path(frames_dir)
    
    # Initialize SDK
    print("Initializing Synchronous SDK...")
    sdk = SyncStreamSDK(cfg_pkl, data_root)
    
    # Setup
    sdk.setup(source_path, frames_dir, **kwargs)
    
    # Load and process audio
    print("Loading audio...")
    audio, sr = librosa.core.load(audio_path, sr=16000)
    num_frames = math.ceil(len(audio) / 16000 * fps)
    print(f"Audio: {len(audio)} samples, {num_frames} frames at {fps}fps")
    
    # Setup frame count
    sdk.setup_Nd(num_frames)
    
    # Convert audio to features
    print("Converting audio to features...")
    audio_features = sdk.wav2feat.wav2feat(audio)
    print(f"Audio features shape: {audio_features.shape}")
    
    # Process audio to motion
    motion_result = sdk.process_audio_to_motion(audio_features)
    
    # Stitch motion
    stitched_motion = sdk.stitch_motion(motion_result)
    
    # Generate all frames
    frame_count = sdk.generate_frames(stitched_motion)
    
    # Create final video
    success = sdk.create_video_with_ffmpeg(frame_count, audio_path, output_path, fps)
    
    if success:
        print(f"✅ Success! Video created: {output_path}")
        
        # Optionally clean up frames
        cleanup_frames = kwargs.get('cleanup_frames', True)
        if cleanup_frames:
            print("Cleaning up frame files...")
            for frame_file in frames_dir.glob("frame_*.png"):
                frame_file.unlink()
            if frames_dir.exists() and not any(frames_dir.iterdir()):
                frames_dir.rmdir()
    else:
        print("❌ Failed to create video")
    
    return success


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None, help="path to model data_root (auto-detected if not provided)")
    parser.add_argument("--cfg_pkl", type=str, default=None, help="path to cfg_pkl (auto-detected if not provided)")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu"], help="force specific device (auto-detected if not provided)")
    parser.add_argument("--print_device_info", action="store_true", help="print device information and exit")

    parser.add_argument("--audio_path", type=str, required=True, help="path to input wav")
    parser.add_argument("--source_path", type=str, required=True, help="path to input image")
    parser.add_argument("--output_path", type=str, required=True, help="path to output mp4")
    parser.add_argument("--frames_dir", type=str, default=None, help="directory to store intermediate frames")
    parser.add_argument("--fps", type=int, default=25, help="output video fps")
    parser.add_argument("--keep_frames", action="store_true", help="keep intermediate frame files")
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

    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Run synchronous inference
    success = run_sync(
        cfg_pkl=cfg_pkl,
        data_root=data_root,
        audio_path=args.audio_path,
        source_path=args.source_path,
        output_path=args.output_path,
        frames_dir=args.frames_dir,
        fps=args.fps,
        cleanup_frames=not args.keep_frames
    )
    
    if not success:
        exit(1) 