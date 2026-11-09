#!/usr/bin/env python3
"""Stream images through YOLO-World and NanoSAM.

YOLO-World supplies open-vocabulary bounding boxes.  Each accepted box is
then used as a box prompt for the TensorRT NanoSAM engines.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import PIL.Image
import torch
from ultralytics import YOLOWorld

from nanosam.utils.predictor import Predictor


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
COLORS = (
    (0, 255, 255),
    (0, 200, 0),
    (255, 120, 0),
    (220, 0, 220),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO-World text detection followed by NanoSAM segmentation"
    )
    parser.add_argument("input", help="An image or a directory of images")
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["step platform", "platform", "loading ramp"],
        help="English YOLO-World class prompts",
    )
    parser.add_argument(
        "--model",
        default="YOLO-World-master/yolov8s-worldv2.pt",
        help="YOLO-World checkpoint",
    )
    parser.add_argument("--image-encoder", default="data/resnet18_image_encoder.engine")
    parser.add_argument("--mask-decoder", default="data/mobile_sam_mask_decoder.engine")
    parser.add_argument("--output-dir", default="data/yolo_world_nanosam")
    parser.add_argument("--conf", type=float, default=0.005)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-det", type=int, default=3)
    parser.add_argument(
        "--class-aware-nms",
        action="store_true",
        help="Keep overlapping boxes from different prompts (off by default for synonym prompts)",
    )
    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=0.45,
        help="Reject boxes covering more than this fraction of the image",
    )
    parser.add_argument(
        "--edge-margin-ratio",
        type=float,
        default=0.0,
        help="Reject boxes whose center is within this fraction of a side",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 processes all images")
    parser.add_argument("--device", default="0")
    parser.add_argument("--cpu", action="store_true", help="Run YOLO-World on CPU")
    parser.add_argument(
        "--tracker",
        default="",
        help="Ultralytics tracker YAML (for example configs/bytetrack_low_conf.yaml)",
    )
    parser.add_argument("--skip-sam", action="store_true", help="Only draw detections")
    parser.add_argument(
        "--save-empty", action="store_true", help="Also save frames with no detections"
    )
    return parser.parse_args()


def collect_images(value):
    path = Path(value)
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    raise FileNotFoundError(f"No image or image directory found: {value}")


def synchronize(device):
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def accepted_detections(result, prompts, width, height, args):
    detections = []
    if result.boxes is None:
        return detections

    image_area = width * height
    margin = args.edge_margin_ratio
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    track_ids = (
        result.boxes.id.int().cpu().numpy()
        if result.boxes.id is not None
        else [None] * len(boxes)
    )
    for box, cls_id, score, track_id in zip(boxes, classes, scores, track_ids):
        x1, y1, x2, y2 = [float(v) for v in box]
        area_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / image_area
        cx = (x1 + x2) / (2 * width)
        cy = (y1 + y2) / (2 * height)
        if area_ratio > args.max_area_ratio:
            continue
        if not (margin <= cx <= 1 - margin and margin <= cy <= 1 - margin):
            continue
        detection = {
            "bbox": [x1, y1, x2, y2],
            "class_id": int(cls_id),
            "label": prompts[int(cls_id)],
            "score": float(score),
        }
        if track_id is not None:
            detection["track_id"] = int(track_id)
        detections.append(detection)
    return detections[: args.max_det]


def segment_boxes(predictor, image, detections):
    synchronize("0")
    started = time.perf_counter()
    predictor.set_image(image)
    synchronize("0")
    encoder_ms = (time.perf_counter() - started) * 1000

    masks = []
    started = time.perf_counter()
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        points = np.asarray([[x1, y1], [x2, y2]], dtype=np.float32)
        labels = np.asarray([2, 3], dtype=np.float32)
        mask, _, _ = predictor.predict(points, labels)
        masks.append((mask[0, 0] > 0).detach().cpu().numpy())
    synchronize("0")
    decoder_ms = (time.perf_counter() - started) * 1000
    return masks, encoder_ms, decoder_ms


def draw_result(image, detections, masks):
    frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR).copy()
    for index, detection in enumerate(detections):
        color_index = detection.get("track_id", index)
        color = COLORS[color_index % len(COLORS)]
        if index < len(masks):
            mask = masks[index]
            frame[mask] = (
                frame[mask].astype(np.float32) * 0.55
                + np.asarray(color, dtype=np.float32) * 0.45
            ).astype(np.uint8)
        x1, y1, x2, y2 = (int(round(v)) for v in detection["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        track_prefix = (
            f"ID {detection['track_id']} " if "track_id" in detection else ""
        )
        text = f"{track_prefix}{detection['label']} {detection['score']:.3f}"
        cv2.putText(
            frame,
            text,
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame


def main():
    args = parse_args()
    paths = collect_images(args.input)
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise RuntimeError("No input images found")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cpu" if args.cpu else args.device
    use_half = device != "cpu"

    detector = YOLOWorld(args.model)
    detector.set_classes(args.prompts)
    sam = None
    if not args.skip_sam:
        sam = Predictor(args.image_encoder, args.mask_decoder)

    # Warm up CUDA and construct the Ultralytics predictor outside the benchmark.
    detector.predict(
        str(paths[0]),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det * 4,
        device=device,
        half=use_half,
        agnostic_nms=not args.class_aware_nms,
        verbose=False,
    )

    rows = []
    run_started = time.perf_counter()
    for index, path in enumerate(paths, start=1):
        image = PIL.Image.open(path).convert("RGB")
        synchronize(device)
        started = time.perf_counter()
        inference_method = detector.track if args.tracker else detector.predict
        inference_kwargs = {}
        if args.tracker:
            inference_kwargs.update(persist=True, tracker=args.tracker)
        result = inference_method(
            str(path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det * 4,
            device=device,
            half=use_half,
            agnostic_nms=not args.class_aware_nms,
            verbose=False,
            **inference_kwargs,
        )[0]
        synchronize(device)
        detector_ms = (time.perf_counter() - started) * 1000

        detections = accepted_detections(
            result, args.prompts, image.width, image.height, args
        )
        masks = []
        encoder_ms = 0.0
        decoder_ms = 0.0
        if detections and sam is not None:
            masks, encoder_ms, decoder_ms = segment_boxes(sam, image, detections)

        if detections or args.save_empty:
            rendered = draw_result(image, detections, masks)
            cv2.imwrite(str(output_dir / f"{path.stem}_result.jpg"), rendered)

        total_ms = detector_ms + encoder_ms + decoder_ms
        rows.append(
            {
                "image": str(path),
                "detections": len(detections),
                "detector_ms": round(detector_ms, 3),
                "sam_encoder_ms": round(encoder_ms, 3),
                "sam_decoder_ms": round(decoder_ms, 3),
                "total_ms": round(total_ms, 3),
                "objects": json.dumps(detections, ensure_ascii=False),
            }
        )
        print(
            f"[{index}/{len(paths)}] {path.name}: detections={len(detections)} "
            f"yolo={detector_ms:.1f}ms sam={encoder_ms + decoder_ms:.1f}ms"
        )

    synchronize(device)
    wall_seconds = time.perf_counter() - run_started
    csv_path = output_dir / "benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    total_model_ms = sum(row["total_ms"] for row in rows)
    detected_frames = sum(row["detections"] > 0 for row in rows)
    track_ids = sorted(
        {
            detection["track_id"]
            for row in rows
            for detection in json.loads(row["objects"])
            if "track_id" in detection
        }
    )
    summary = {
        "images": len(rows),
        "detected_frames": detected_frames,
        "prompts": args.prompts,
        "tracker": args.tracker or None,
        "unique_track_ids": track_ids,
        "model_fps": round(1000 * len(rows) / total_model_ms, 3),
        "wall_fps": round(len(rows) / wall_seconds, 3),
        "average_model_ms": round(total_model_ms / len(rows), 3),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
