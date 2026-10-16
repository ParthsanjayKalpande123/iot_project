"""CLI runner for the conveyor vision module."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import yaml

from vision import (
    WARNING_EVENTS,
    CameraSource,
    Detector,
    FrameSource,
    VideoSource,
    WebcamSource,
)

LOGGER = logging.getLogger("conveyor_vision")


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root in {path} must be a YAML mapping")
    if config.get("source") not in {"video", "camera"}:
        raise ValueError('config.source must be either "video" or "camera"')
    base = path.resolve().parent
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("config.model must be a YAML mapping")
    for key in ("path", "labels_path"):
        model[key] = str(_resolve_path(base, str(model[key])))
    config["output_dir"] = str(_resolve_path(base, str(config.get("output_dir", "output"))))
    if config["source"] == "video":
        config["video_path"] = str(_resolve_path(base, str(config["video_path"])))
    return config


def make_source(config: Mapping[str, Any], realtime: bool) -> FrameSource:
    if config["source"] == "video":
        return VideoSource(config["video_path"], realtime=realtime)
    camera = config.get("camera", {})
    if camera.get("backend", "picamera2") == "webcam":
        return WebcamSource(
            camera_index=int(camera.get("index", 0)),
            width=int(camera.get("width", 1280)),
            height=int(camera.get("height", 720)),
        )
    if camera.get("backend", "picamera2") != "picamera2":
        raise ValueError('camera.backend must be "picamera2" or "webcam"')
    return CameraSource(
        width=int(camera.get("width", 1280)),
        height=int(camera.get("height", 720)),
        camera_index=int(camera.get("index", 0)),
    )


def _write_calibrated_config(config_path: Path, measurements: list[tuple[float, float]],
                             margin: float) -> tuple[float, float]:
    if not measurements:
        raise ValueError(
            "No bottle detections were collected. Check the model, labels, confidence "
            "threshold, ROI, and calibration segment."
        )
    if margin < 0:
        raise ValueError("calibration_safety_margin must be non-negative")
    areas = [area for area, _ in measurements]
    heights = [height for _, height in measurements]
    area_min, area_max = min(areas), max(areas)
    height_min, height_max = min(heights), max(heights)
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    config["size_calibration"] = {
        "calibrated": True,
        "min_area_ratio": max(0.0, area_min * (1.0 - margin)),
        "max_area_ratio": area_max * (1.0 + margin),
        "min_height_ratio": max(0.0, height_min * (1.0 - margin)),
        "max_height_ratio": height_max * (1.0 + margin),
        "safety_margin": margin,
        "samples": len(measurements),
    }
    with config_path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file, sort_keys=False, allow_unicode=False)
    return area_max, height_max


def run_calibration(
    config: Mapping[str, Any],
    config_path: Path,
    start_s: float | None,
    end_s: float | None,
    realtime: bool,
) -> int:
    if start_s is not None and start_s < 0:
        raise ValueError("--calib-start must be non-negative")
    if end_s is not None and end_s < 0:
        raise ValueError("--calib-end must be non-negative")
    if start_s is not None and end_s is not None and end_s <= start_s:
        raise ValueError("--calib-end must be greater than --calib-start")
    if config["source"] == "camera" and (start_s is not None or end_s is not None):
        raise ValueError("--calib-start and --calib-end apply only to video sources")

    detector = Detector(config)
    measurements: list[tuple[float, float]] = []
    max_camera_frames = 30
    with make_source(config, realtime=realtime) as source:
        camera_frames = 0
        while True:
            packet = source.read()
            if packet is None:
                break
            if start_s is not None and packet.timestamp_s < start_s:
                continue
            if end_s is not None and packet.timestamp_s > end_s:
                break
            measurements.extend(detector.calibration_measurements(packet.frame))
            if config["source"] == "camera":
                camera_frames += 1
                if camera_frames >= max_camera_frames:
                    break

    margin = float(config.get("calibration_safety_margin", 0.15))
    area_max, height_max = _write_calibrated_config(config_path, measurements, margin)
    print(
        f"Calibration saved to {config_path}: {len(measurements)} bottle detections; "
        f"observed max area_ratio={area_max:.6f}, height_ratio={height_max:.6f}; "
        f"safety margin={margin:.1%}."
    )
    return 0


def _read_intervals(path: Path | None) -> list[tuple[float, float, str]]:
    if path is None:
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"start_s", "end_s", "expected_event"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(
                "Ground-truth CSV must have start_s,end_s,expected_event columns"
            )
        intervals = []
        for row in reader:
            start, end = float(row["start_s"]), float(row["end_s"])
            event = row["expected_event"].strip().upper()
            if start < 0 or end < start or not event:
                raise ValueError(f"Invalid ground-truth row: {row}")
            intervals.append((start, end, event))
        return intervals


def _print_evaluation(
    intervals: list[tuple[float, float, str]],
    stats: dict[str, dict[str, int]],
) -> None:
    events = sorted(set(WARNING_EVENTS) | {event for _, _, event in intervals})
    print("\nPer-event evaluation (frame-level):")
    print(f"{'Event':24} {'Precision':>10} {'Recall':>10} {'TP':>7} {'FP':>7} {'FN':>7}")
    for event in events:
        counts = stats.get(event, {"tp": 0, "fp": 0, "fn": 0})
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        print(f"{event:24} {precision:10.3f} {recall:10.3f} {tp:7d} {fp:7d} {fn:7d}")


def run_analysis(
    config: Mapping[str, Any],
    realtime: bool,
    show: bool,
    ground_truth_path: Path | None,
) -> int:
    detector = Detector(config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.csv"
    annotated_path = output_dir / "output_annotated.mp4"
    intervals = _read_intervals(ground_truth_path)
    evaluation_stats: dict[str, dict[str, int]] = {}
    for _, _, event in intervals:
        evaluation_stats.setdefault(event, {"tp": 0, "fp": 0, "fn": 0})

    sample_interval = float(config.get("video_sample_interval_s", 0.0))
    if sample_interval < 0:
        raise ValueError("video_sample_interval_s must be non-negative")
    source = make_source(config, realtime=realtime)
    source_fps = getattr(source, "fps", float(config.get("camera", {}).get("output_fps", 10.0)))
    output_fps = 1.0 / sample_interval if config["source"] == "video" and sample_interval > 0 \
        else float(source_fps)
    if not np.isfinite(output_fps) or output_fps <= 0:
        output_fps = 10.0

    frame_count = 0
    warning_counts: dict[str, int] = {}
    previous_warning: str | None = None
    next_sample_s = 0.0
    latencies: deque[float] = deque(maxlen=100_000)
    latency_rng = random.Random(0)
    latency_seen = 0
    writer: cv2.VideoWriter | None = None
    try:
        with source, results_path.open("w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "video_time_s", "status", "event", "labels", "scores",
                "area_ratio", "inference_ms",
            ])
            while True:
                packet = source.read()
                if packet is None:
                    break
                if config["source"] == "video" and sample_interval > 0:
                    if packet.timestamp_s + 1e-9 < next_sample_s:
                        continue
                    next_sample_s = packet.timestamp_s + sample_interval

                result = detector.analyze_frame(packet.frame)
                annotated_frame = detector.annotate_frame(packet.frame, result)
                if writer is None:
                    height, width = packet.frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(annotated_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        output_fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise OSError(f"Could not create annotated video: {annotated_path}")
                writer.write(annotated_frame)
                frame_count += 1
                inference_ms = float(result["inference_ms"])
                latency_seen += 1
                if len(latencies) < latencies.maxlen:
                    latencies.append(inference_ms)
                else:
                    replacement = latency_rng.randrange(latency_seen)
                    if replacement < latencies.maxlen:
                        latencies[replacement] = inference_ms

                detections = result["detections"]
                csv_writer.writerow([
                    f"{packet.timestamp_s:.6f}",
                    result["status"],
                    result["event"],
                    json.dumps([item["label"] for item in detections]),
                    json.dumps([item["score"] for item in detections]),
                    json.dumps([item["area_ratio"] for item in detections]),
                    f"{inference_ms:.3f}",
                ])
                is_warning = result["status"] == "WARNING"
                current_warning = str(result["event"]) if is_warning else None
                if current_warning and current_warning != previous_warning:
                    warning_counts[current_warning] = warning_counts.get(current_warning, 0) + 1
                previous_warning = current_warning

                if intervals:
                    timestamp = packet.timestamp_s
                    expected = next(
                        (event for start, end, event in intervals
                         if start <= timestamp <= end),
                        "NORMAL",
                    )
                    predicted = result["event"] if is_warning else "NORMAL"
                    event_set = set(WARNING_EVENTS) | {expected, predicted}
                    for event in event_set - {"NORMAL"}:
                        counts = evaluation_stats.setdefault(
                            event, {"tp": 0, "fp": 0, "fn": 0}
                        )
                        truth = expected == event
                        prediction = predicted == event
                        counts["tp"] += int(truth and prediction)
                        counts["fp"] += int(not truth and prediction)
                        counts["fn"] += int(truth and not prediction)

                if show:
                    cv2.imshow("Conveyor vision", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    if not frame_count:
        raise RuntimeError("No frames were analyzed; check the selected input source.")
    p95 = float(np.percentile(np.asarray(latencies), 95))
    average = sum(latencies) / len(latencies)
    print(f"Frames analyzed: {frame_count}")
    print("WARNING episodes by type:")
    if warning_counts:
        for event, count in sorted(warning_counts.items()):
            print(f"  {event}: {count}")
    else:
        print("  none")
    print(
        f"Inference time: average={average:.2f} ms, p95={p95:.2f} ms "
        f"(p95 uses a reservoir sample of at most 100,000 frames)."
    )
    print(f"Annotated video: {annotated_path}")
    print(f"Frame results: {results_path}")
    if intervals:
        _print_evaluation(intervals, evaluation_stats)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run offline TFLite conveyor-belt object monitoring."
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--realtime", action="store_true",
                        help="Play a video at its original source timing.")
    parser.add_argument("--show", action="store_true",
                        help="Display annotated preview (requires a desktop display).")
    parser.add_argument("--calibrate", action="store_true",
                        help="Calibrate bottle size thresholds using normal 1 L bottles.")
    parser.add_argument("--calib-start", type=float,
                        help="Video calibration segment start time in seconds.")
    parser.add_argument("--calib-end", type=float,
                        help="Video calibration segment end time in seconds.")
    parser.add_argument("--ground-truth", type=Path,
                        help="Optional labels.csv with manually labeled event intervals.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        config_path = args.config.resolve()
        config = load_config(config_path)
        if args.calibrate:
            return run_calibration(
                config, config_path, args.calib_start, args.calib_end, args.realtime
            )
        if args.calib_start is not None or args.calib_end is not None:
            parser.error("--calib-start and --calib-end require --calibrate")
        ground_truth = args.ground_truth.resolve() if args.ground_truth else None
        return run_analysis(config, args.realtime, args.show, ground_truth)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, KeyError,
            yaml.YAMLError) as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
