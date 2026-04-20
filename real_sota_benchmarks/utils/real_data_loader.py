"""
real_data_loader.py
===================
Core: uses cv2 to read real sample.mp4, dynamically extracts video frames and text prompt
based on 4K, 8K, 12K, 16K token targets, and computes real token counts via AutoProcessor.
"""

import os
import cv2
import tempfile
import torch
import numpy as np
from typing import Dict, Any, List
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

SAMPLE_VIDEO = os.environ.get("BENCHMARK_VIDEO", "sample.mp4")


def _find_sample_video() -> str:
    """Locate an available test video."""
    if os.path.exists(SAMPLE_VIDEO):
        return os.path.abspath(SAMPLE_VIDEO)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    candidates = [
        os.path.join(project_root, "dummy_1min.mp4"),
        os.path.join(project_root, "test_video.mp4"),
        os.path.join(project_root, "dummy_2min.mp4"),
        os.path.join(project_root, "dummy_4min.mp4"),
        os.path.join(project_root, "dummy_8min.mp4"),
        os.path.join(project_root, "needle_video_5min.mp4"),
        os.path.join(project_root, "needle_video_10min.mp4"),
        os.path.join(project_root, "needle_video_15min.mp4"),
        os.path.join(project_root, "needle_video_20min.mp4"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"No available video found. Please place sample.mp4 in {os.getcwd()}")


def extract_frames_cv2(video_path: str, num_frames: int) -> List[str]:
    """
    Use OpenCV to uniformly extract num_frames frames from the video,
    save as temporary JPEGs, and return the path list.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Cannot read video frame count: {video_path}")

    indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames)).astype(int)
    frame_paths = []
    temp_dir = tempfile.mkdtemp(prefix="benchmark_frames_")

    for idx, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        path = os.path.join(temp_dir, f"frame_{idx:04d}.jpg")
        pil_img.save(path, quality=95)
        frame_paths.append(path)

    cap.release()
    if len(frame_paths) == 0:
        raise ValueError(f"Failed to extract any frames from video: {video_path}")
    return frame_paths


def _estimate_tokens_per_frame(processor: AutoProcessor, frame_path: str) -> int:
    """
    Estimate visual tokens per frame via Processor (marginal increment).
    """
    messages_1 = [
        {"role": "user", "content": [{"type": "video", "video": [frame_path]}, {"type": "text", "text": "."}]}
    ]
    text_1 = processor.apply_chat_template(messages_1, tokenize=False, add_generation_prompt=True)
    image_inputs_1, video_inputs_1 = process_vision_info(messages_1)
    inputs_1 = processor(text=[text_1], images=image_inputs_1, videos=video_inputs_1, padding=True, return_tensors="pt")
    tokens_1 = int(inputs_1.input_ids.shape[1])

    messages_2 = [
        {"role": "user", "content": [{"type": "video", "video": [frame_path, frame_path]}, {"type": "text", "text": "."}]}
    ]
    text_2 = processor.apply_chat_template(messages_2, tokenize=False, add_generation_prompt=True)
    image_inputs_2, video_inputs_2 = process_vision_info(messages_2)
    inputs_2 = processor(text=[text_2], images=image_inputs_2, videos=video_inputs_2, padding=True, return_tensors="pt")
    tokens_2 = int(inputs_2.input_ids.shape[1])

    marginal = max(tokens_2 - tokens_1, 1)
    return marginal


def build_multimodal_inputs(
    target_tokens: int,
    processor: AutoProcessor,
    text_prompt: str = "Please describe the video in detail.",
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Dynamically extract real video frames based on target_tokens goal,
    construct real multimodal inputs.
    Returns a dict containing actual token count, input tensors, messages, etc.
    """
    video_path = _find_sample_video()

    # 1. Estimate tokens per frame
    temp_frames = extract_frames_cv2(video_path, 2)
    tokens_per_frame = _estimate_tokens_per_frame(processor, temp_frames[0])

    # 2. Compute text prompt token count
    text_only_messages = [{"role": "user", "content": [{"type": "text", "text": text_prompt}]}]
    text_only_str = processor.apply_chat_template(text_only_messages, tokenize=False, add_generation_prompt=True)
    text_only_inputs = processor(text=[text_only_str], padding=True, return_tensors="pt")
    text_tokens = int(text_only_inputs.input_ids.shape[1])

    # 3. Compute number of frames to extract
    target_visual_tokens = max(target_tokens - text_tokens, 100)
    num_frames = max(1, int(target_visual_tokens // tokens_per_frame))

    # 4. Limit frames to total video frames and max 2000 (avoid excessive I/O)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    num_frames = min(num_frames, total_frames, 2000)

    # 5. Extract frames
    frame_paths = extract_frames_cv2(video_path, num_frames)

    # 6. Construct messages and compute actual token count
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frame_paths},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    actual_tokens = int(inputs.input_ids.shape[1])
    inputs = {k: v.to(device) for k, v in inputs.items()}

    return {
        "inputs": inputs,
        "actual_tokens": actual_tokens,
        "num_frames": num_frames,
        "video_path": video_path,
        "text_prompt": text_prompt,
        "frame_paths": frame_paths,
        "messages": messages,
    }
