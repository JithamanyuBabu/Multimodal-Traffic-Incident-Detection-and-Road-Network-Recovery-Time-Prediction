"""
Computer Vision Pipeline for Traffic Incident & Scene Evidence Detection (Module B)
Uses OpenCV (cv2) and Pillow (PIL) to inspect uploaded images and sampled video frames,
extract multi-scale vehicle/obstacle regions via asphalt-background contrast & gradient clustering,
evaluate collision/debris/congestion evidence, and produce structured visual embeddings + bounding boxes.
"""
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB limit
MAX_VIDEO_DURATION_SEC = 60.0
CONFIDENCE_THRESHOLD = 0.52
CV_MODEL_VERSION = "OpenCV-Multiscale-TrafficScene-v2.1"


def validate_media_upload(filename: str, content: bytes) -> Tuple[str, str]:
    """
    Validates file extension, size limit, and magic bytes / decodability.
    Returns (media_type, extension) where media_type is 'image' or 'video'.
    Raises ValueError on invalid, empty, oversized, or corrupt uploads.
    """
    if not filename or not content:
        raise ValueError("Empty file upload or missing filename.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(f"File size ({len(content)} bytes) exceeds limit of {MAX_UPLOAD_BYTES} bytes.")

    ext = Path(filename).suffix.lower()
    if ext in SUPPORTED_IMAGE_EXTENSIONS:
        try:
            img = Image.open(io.BytesIO(content))
            img.verify()
        except Exception as exc:
            raise ValueError(f"Corrupt or invalid image file: {exc}") from exc
        return "image", ext
    elif ext in SUPPORTED_VIDEO_EXTENSIONS:
        if len(content) < 64:
            raise ValueError("Corrupt or truncated video file.")
        return "video", ext
    else:
        raise ValueError(
            f"Unsupported file format '{ext}'. Supported formats: "
            f"{sorted(SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS)}"
        )


def _non_max_suppression(boxes: List[Dict[str, Any]], iou_thresh: float = 0.30) -> List[Dict[str, Any]]:
    if not boxes:
        return []
    boxes_sorted = sorted(boxes, key=lambda b: b["confidence"], reverse=True)
    kept: List[Dict[str, Any]] = []
    for cand in boxes_sorted:
        x1, y1, w1, h1 = cand["bbox"]
        area1 = max(1e-5, float(w1 * h1))
        overlap = False
        for k in kept:
            x2, y2, w2, h2 = k["bbox"]
            area2 = max(1e-5, float(w2 * h2))
            ix1 = max(x1, x2)
            iy1 = max(y1, y2)
            ix2 = min(x1 + w1, x2 + w2)
            iy2 = min(y1 + h1, y2 + h2)
            inter = max(0.0, float(ix2 - ix1)) * max(0.0, float(iy2 - iy1))
            iou = inter / (area1 + area2 - inter + 1e-6)
            if iou > iou_thresh:
                overlap = True
                break
        if not overlap:
            kept.append(cand)
    return kept


def analyze_single_frame(bgr_frame: np.ndarray, timestamp_sec: float = 0.0) -> Dict[str, Any]:
    """
    Performs multi-cue computer vision analysis on a BGR traffic frame:
      1. Asphalt background deviation & multi-scale morphological contour extraction for individual vehicles.
      2. Center-lane transverse vehicle collision signature & dense upstream queue tailback analysis.
      3. Stalled commercial truck + scattered road-surface cargo/debris signature.
      4. Wet-asphalt specular highlight detection.
    """
    h, w = bgr_frame.shape[:2]
    if h < 32 or w < 32:
        raise ValueError(f"Frame resolution ({w}x{h}) is too small for reliable incident analysis.")

    target_w = 640
    scale = target_w / float(w)
    target_h = max(64, int(round(h * scale)))
    resized = cv2.resize(bgr_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # ROI covering roadway (exclude top 18% sky/gantry banner)
    roi_y0 = int(target_h * 0.18)
    roi_gray = gray[roi_y0:, :]
    roi_hsv = hsv[roi_y0:, :]
    roi_bgr = resized[roi_y0:, :]
    rh, rw = roi_gray.shape[:2]

    # Estimate road asphalt baseline intensity & saturation
    asphalt_med = float(np.median(roi_gray))
    diff_from_road = cv2.absdiff(roi_gray, np.full_like(roi_gray, int(asphalt_med)))

    # Sobel gradients
    gx = cv2.Sobel(roi_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    grad_u8 = np.clip(grad_mag, 0, 255).astype(np.uint8)

    # Color saturation mask (vehicles, brake lights, hazard markers, cardboard/debris)
    sat_mask = (roi_hsv[:, :, 1] > 48).astype(np.uint8) * 255
    bright_mask = (diff_from_road > 36).astype(np.uint8) * 255
    edge_mask = (grad_u8 > 58).astype(np.uint8) * 255

    combined = cv2.bitwise_or(bright_mask, sat_mask)
    combined = cv2.bitwise_and(combined, cv2.dilate(edge_mask, np.ones((3, 3), np.uint8), iterations=1))

    # Morphological closing with moderate kernel to keep individual vehicles distinct
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes: List[Dict[str, Any]] = []
    roi_area = float(rh * rw)

    # Warm / orange-brown hue mask for cardboard / cargo debris or hazard cones
    hue = roi_hsv[:, :, 0]
    sat = roi_hsv[:, :, 1]
    val = roi_hsv[:, :, 2]
    warm_debris_mask = ((hue >= 8) & (hue <= 28) & (sat >= 55) & (val >= 90) & (val <= 235)).astype(np.uint8)
    warm_ratio_global = float(np.mean(warm_debris_mask))

    # Specular reflection mask on road surface (high brightness, low saturation)
    specular_mask = ((sat < 35) & (val > 205)).astype(np.uint8)
    wet_reflection_ratio = float(np.mean(specular_mask))

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        box_area = float(cw * ch)
        area_frac = box_area / roi_area
        if area_frac < 0.004 or area_frac > 0.32:
            continue
        aspect = cw / float(max(1, ch))
        if aspect < 0.35 or aspect > 4.2:
            continue

        patch_edge = float(np.mean(grad_u8[y : y + ch, x : x + cw])) / 255.0
        patch_warm = float(np.mean(warm_debris_mask[y : y + ch, x : x + cw]))
        center_x = (x + 0.5 * cw) / float(rw)
        center_y = (y + 0.5 * ch) / float(rh)

        # Classify candidate region
        if area_frac < 0.022 and patch_warm > 0.16 and center_y > 0.32:
            label = "road_debris_hazard"
            conf = min(0.94, 0.58 + patch_warm * 0.9 + patch_edge * 0.3)
        elif area_frac > 0.065 and (aspect < 1.35 or ch > rh * 0.34):
            label = "heavy_truck_or_bus"
            conf = min(0.95, 0.64 + patch_edge * 0.45 + min(0.18, area_frac))
        elif 0.28 <= center_x <= 0.72 and 0.30 <= center_y <= 0.82 and aspect > 1.35 and patch_edge > 0.24:
            label = "damaged_or_stalled_vehicle"
            conf = min(0.96, 0.66 + patch_edge * 0.5)
        else:
            label = "vehicle"
            conf = min(0.93, 0.56 + patch_edge * 0.45 + min(0.15, area_frac * 2.0))

        if conf >= CONFIDENCE_THRESHOLD:
            raw_boxes.append({
                "label": label,
                "confidence": round(float(conf), 3),
                "bbox": [
                    round(x / float(rw), 4),
                    round((y + roi_y0) / float(target_h), 4),
                    round(cw / float(rw), 4),
                    round(ch / float(target_h), 4),
                ],
                "area_fraction": round(area_frac, 4),
            })

    boxes = _non_max_suppression(raw_boxes, iou_thresh=0.30)[:14]

    # Compute structured scene scores
    upper_queue_zone = grad_u8[: int(rh * 0.48), :]
    lower_incident_zone = grad_u8[int(rh * 0.35) : int(rh * 0.88), int(rw * 0.20) : int(rw * 0.80)]

    upper_edge_density = float(np.mean(upper_queue_zone > 55))
    center_edge_density = float(np.mean(lower_incident_zone > 60))

    num_vehicles = sum(1 for b in boxes if "vehicle" in b["label"] or "truck" in b["label"])
    num_debris = sum(1 for b in boxes if b["label"] == "road_debris_hazard")
    num_stalled = sum(1 for b in boxes if b["label"] in ("damaged_or_stalled_vehicle", "heavy_truck_or_bus"))

    # Queue density reflects upper roadway occupancy + total detected vehicles
    vis_queue_density = float(np.clip(upper_edge_density * 1.65 + min(0.45, num_vehicles * 0.045), 0.05, 0.98))

    # Debris score reflects scattered small obstacles + warm cargo signature + lower-lane clutter
    vis_debris_score = float(
        np.clip(
            num_debris * 0.22 + warm_ratio_global * 3.8 + (0.20 if num_stalled >= 1 and warm_ratio_global > 0.04 else 0.0),
            0.02,
            0.96,
        )
    )

    # Collision score reflects central lane blockage + high multi-vehicle density + wet/angled anomaly
    center_boxes = [
        b for b in boxes if 0.25 <= (b["bbox"][0] + 0.5 * b["bbox"][2]) <= 0.75 and b["bbox"][1] >= 0.35
    ]
    vis_collision_score = float(
        np.clip(
            center_edge_density * 1.35
            + len(center_boxes) * 0.11
            + (0.18 if vis_queue_density > 0.52 and len(center_boxes) >= 2 else 0.0)
            - vis_debris_score * 0.25,
            0.03,
            0.97,
        )
    )

    stopped_vehicles = int(np.clip(len(center_boxes) if vis_collision_score > 0.48 or vis_debris_score > 0.45 else 0, 0, 5))

    # Determine visual incident classification & human-in-the-loop verification status
    if vis_debris_score >= 0.52 and vis_debris_score >= vis_collision_score - 0.08:
        visual_category = "hazard_debris"
        scene_conf = round(min(0.95, 0.58 + vis_debris_score * 0.38), 3)
    elif vis_collision_score >= 0.56 and stopped_vehicles >= 2:
        visual_category = "collision_multi_vehicle"
        scene_conf = round(min(0.96, 0.60 + vis_collision_score * 0.36), 3)
    elif vis_collision_score >= 0.46 or stopped_vehicles == 1:
        visual_category = "disabled_vehicle"
        scene_conf = round(min(0.91, 0.55 + vis_collision_score * 0.32), 3)
    elif vis_queue_density >= 0.68:
        visual_category = "secondary_shockwave"
        scene_conf = round(min(0.89, 0.54 + vis_queue_density * 0.32), 3)
    else:
        visual_category = "normal_flow"
        scene_conf = round(min(0.94, 0.72 + (1.0 - max(vis_collision_score, vis_debris_score)) * 0.20), 3)

    if visual_category == "normal_flow":
        verification_status = "verified_normal_flow"
    elif scene_conf >= 0.75:
        verification_status = "potential_incident_detected"
    else:
        verification_status = "low_confidence_requires_operator_review"

    return {
        "timestamp_sec": round(float(timestamp_sec), 2),
        "frame_resolution": {"width": w, "height": h},
        "predicted_visual_category": visual_category,
        "confidence": scene_conf,
        "verification_status": verification_status,
        "detected_objects_count": len(boxes),
        "bounding_boxes": boxes,
        "visual_features": {
            "vis_collision_score": round(vis_collision_score, 3),
            "vis_debris_score": round(vis_debris_score, 3),
            "vis_stopped_vehicles": stopped_vehicles,
            "vis_queue_density": round(vis_queue_density, 3),
            "vis_wet_surface_indicator": round(wet_reflection_ratio, 3),
            "vis_confidence": scene_conf,
        },
        "model_version": CV_MODEL_VERSION,
    }


def process_media_bytes(filename: str, content: bytes, frame_sample_fps: float = 1.0) -> Dict[str, Any]:
    """
    Processes an uploaded image or video byte buffer through the OpenCV pipeline.
    Returns frame-level detections, aggregated visual feature vector, and status metadata.
    Uses Windows-compatible closed temporary file handles (`tempfile.mkstemp`) for video containers.
    """
    media_type, ext = validate_media_upload(filename, content)

    if media_type == "image":
        arr = np.frombuffer(content, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("OpenCV failed to decode image buffer.")
        frame_res = analyze_single_frame(bgr, timestamp_sec=0.0)
        return {
            "media_type": "image",
            "filename": filename,
            "frames_analyzed": 1,
            "duration_sec": 0.0,
            "primary_result": frame_res,
            "frame_timeline": [frame_res],
            "processing_status": "completed",
        }

    fd, tmp_path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    timeline: List[Dict[str, Any]] = []
    duration_sec = 0.0
    try:
        Path(tmp_path).write_bytes(content)
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Unable to open video container.")
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
            if fps <= 0 or fps > 240:
                fps = 15.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration_sec = total_frames / fps if total_frames > 0 else 0.0
            if duration_sec > MAX_VIDEO_DURATION_SEC:
                raise ValueError(
                    f"Video duration ({duration_sec:.1f}s) exceeds maximum allowed ({MAX_VIDEO_DURATION_SEC}s)."
                )
            stride = max(1, int(round(fps / max(0.5, frame_sample_fps))))
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                if frame_idx % stride == 0:
                    ts = frame_idx / fps
                    timeline.append(analyze_single_frame(frame, timestamp_sec=ts))
                frame_idx += 1
                if len(timeline) >= 30:
                    break
        finally:
            cap.release()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    if not timeline:
        raise ValueError("No valid video frames could be decoded from video clip.")

    primary = max(
        timeline,
        key=lambda r: (
            r["visual_features"]["vis_collision_score"]
            + r["visual_features"]["vis_debris_score"]
            + 0.3 * r["visual_features"]["vis_queue_density"]
        ),
    )
    return {
        "media_type": "video",
        "filename": filename,
        "frames_analyzed": len(timeline),
        "duration_sec": round(duration_sec, 2),
        "primary_result": primary,
        "frame_timeline": timeline,
        "processing_status": "completed",
    }
