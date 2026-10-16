"""On-device TFLite conveyor-belt object detection and event classification."""

from __future__ import annotations

import csv
import logging
import math
import re
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)
WARNING_EVENTS = {
    "OVERSIZED_BOTTLE",
    "UNDERSIZED_BOTTLE",
    "FOREIGN_OBJECT",
    "MULTIPLE_OBJECTS",
}


def classify_bottle_size(
    box: Sequence[float],
    frame_width: int,
    frame_height: int,
    max_area_ratio: float,
    max_height_ratio: float,
    min_area_ratio: float = 0.0,
    min_height_ratio: float = 0.0,
) -> str:
    """Classify an xyxy pixel box against calibrated normalized size limits.

    This function is deliberately independent of OpenCV, a camera, and a model.
    Oversized means either dimension exceeds its upper bound; undersized means
    both dimensions are below their lower bounds. That conjunction avoids
    rejecting normal bottles based on one unusually narrow/tall box.
    """
    if len(box) != 4:
        raise ValueError("box must contain x1, y1, x2, y2")
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    if min_area_ratio < 0 or min_height_ratio < 0:
        raise ValueError("size thresholds must be non-negative")
    if max_area_ratio < min_area_ratio or max_height_ratio < min_height_ratio:
        raise ValueError("size thresholds must be non-negative")

    x1, y1, x2, y2 = (float(value) for value in box)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area_ratio = (width * height) / (frame_width * frame_height)
    height_ratio = height / frame_height
    if area_ratio > max_area_ratio or height_ratio > max_height_ratio:
        return "OVERSIZED_BOTTLE"
    if area_ratio < min_area_ratio and height_ratio < min_height_ratio:
        return "UNDERSIZED_BOTTLE"
    return "NORMAL"


@dataclass(frozen=True)
class FramePacket:
    """A BGR frame and its time offset in the source stream, in seconds."""

    frame: np.ndarray
    timestamp_s: float


class FrameSource(ABC):
    """Common interface for camera and video inputs."""

    @abstractmethod
    def read(self) -> FramePacket | None:
        """Return the next frame, or None when the source is exhausted."""

    @abstractmethod
    def close(self) -> None:
        """Release source resources."""

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class VideoSource(FrameSource):
    """Read frames from a video file, optionally pacing them to source time."""

    def __init__(self, video_path: str | Path, realtime: bool = False) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {self.video_path}")
        self.capture = cv2.VideoCapture(str(self.video_path))
        if not self.capture.isOpened():
            self.capture.release()
            raise OSError(f"Could not open video file: {self.video_path}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(self.fps) or self.fps <= 0:
            self.fps = 30.0
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.realtime = realtime
        self._frame_index = 0
        self._last_timestamp_s: float | None = None
        self._wall_start: float | None = None

    def read(self) -> FramePacket | None:
        ok, frame = self.capture.read()
        if not ok:
            return None
        timestamp_s = float(self.capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        if (not math.isfinite(timestamp_s) or timestamp_s < 0
                or (self._last_timestamp_s is not None
                    and timestamp_s <= self._last_timestamp_s)):
            minimum_timestamp = (
                0.0 if self._last_timestamp_s is None
                else self._last_timestamp_s + 1.0 / self.fps
            )
            timestamp_s = max(self._frame_index / self.fps, minimum_timestamp)
        self._last_timestamp_s = timestamp_s
        self._frame_index += 1

        if self.realtime:
            if self._wall_start is None:
                self._wall_start = time.perf_counter() - timestamp_s
            delay_s = self._wall_start + timestamp_s - time.perf_counter()
            if delay_s > 0:
                time.sleep(delay_s)
        return FramePacket(frame=frame, timestamp_s=timestamp_s)

    def close(self) -> None:
        self.capture.release()


class CameraSource(FrameSource):
    """Capture BGR frames from a Raspberry Pi Camera using Picamera2."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        camera_index: int = 0,
    ) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "Picamera2 is unavailable. Install it on Raspberry Pi OS or use "
                "source: video in config.yaml."
            ) from exc

        try:
            self.camera = Picamera2(camera_num=camera_index)
            config = self.camera.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"}
            )
            self.camera.configure(config)
            self.camera.start()
        except Exception as exc:
            raise RuntimeError(f"Could not initialize Picamera2: {exc}") from exc
        self._started_at = time.perf_counter()

    def read(self) -> FramePacket:
        try:
            rgb_frame = self.camera.capture_array()
        except Exception as exc:
            raise RuntimeError(f"Picamera2 frame capture failed: {exc}") from exc
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        return FramePacket(frame=frame, timestamp_s=time.perf_counter() - self._started_at)

    def close(self) -> None:
        self.camera.stop()
        self.camera.close()


class WebcamSource(FrameSource):
    """Capture BGR frames from a USB or built-in webcam via OpenCV."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        self.capture = (
            cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if sys.platform == "win32"
            else cv2.VideoCapture(camera_index)
        )
        if not self.capture.isOpened():
            self.capture.release()
            raise OSError(
                f"Could not open webcam at index {camera_index}. "
                "Check that it is connected and not in use by another application."
            )
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(self.fps) or self.fps <= 0:
            self.fps = 30.0
        self._started_at = time.perf_counter()

    def read(self) -> FramePacket:
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError("Could not read a frame from the webcam.")
        return FramePacket(
            frame=frame,
            timestamp_s=time.perf_counter() - self._started_at,
        )

    def close(self) -> None:
        self.capture.release()


def _load_labels(path: str | Path, index_offset: int) -> dict[int, str]:
    labels: dict[int, str] = {}
    label_path = Path(path)
    if not label_path.is_file():
        raise FileNotFoundError(f"Labels file does not exist: {label_path}")

    indexed_line = re.compile(r"^\s*(\d+)\s+(.+?)\s*$")
    plain_index = 0
    with label_path.open("r", encoding="utf-8") as labels_file:
        for raw_line in labels_file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            match = indexed_line.match(line)
            if match:
                labels[int(match.group(1))] = match.group(2)
            else:
                labels[plain_index + index_offset] = line
                plain_index += 1
    if not labels:
        raise ValueError(f"No labels found in {label_path}")
    return labels


class Detector:
    """TFLite detector plus ROI filtering, size checks, and warning debounce."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        model_cfg = config["model"]
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter  # type: ignore[no-redef]
            except ImportError as exc:
                raise RuntimeError(
                    "No TFLite interpreter found. Install ai-edge-litert or "
                    "tflite-runtime for this Python environment."
                ) from exc

        model_path = Path(str(model_cfg["path"]))
        if not model_path.is_file():
            raise FileNotFoundError(f"TFLite model does not exist: {model_path}")
        try:
            self.interpreter = Interpreter(model_path=str(model_path), num_threads=4)
            self.interpreter.allocate_tensors()
        except Exception as exc:
            raise RuntimeError(f"Could not load TFLite model {model_path}: {exc}") from exc

        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()
        if len(self.output_details) < 4:
            raise ValueError(
                "Expected at least four TFLite output tensors: boxes, classes, "
                "scores, count."
            )
        input_shape = self.input_detail["shape"]
        if len(input_shape) != 4 or int(input_shape[0]) != 1:
            raise ValueError(f"Expected a batch-1 NHWC model input, got {input_shape}")
        self.input_height = int(input_shape[1])
        self.input_width = int(input_shape[2])
        if int(input_shape[3]) != 3:
            raise ValueError(f"Expected three input channels, got shape {input_shape}")

        self.output_order = tuple(model_cfg.get(
            "output_order", ["boxes", "classes", "scores", "count"]
        ))
        if len(self.output_order) != 4 or set(self.output_order) != {
            "boxes", "classes", "scores", "count"
        }:
            raise ValueError(
                "model.output_order must contain boxes, classes, scores, count once each"
            )
        self.labels = _load_labels(
            model_cfg["labels_path"], int(model_cfg.get("label_index_offset", 0))
        )
        self._input_color_order = str(model_cfg.get("input_color_order", "RGB"))
        self._float_input_scale = float(model_cfg.get("float_input_scale", 1.0 / 255.0))
        self._integer_input_scale = float(model_cfg.get("integer_input_scale", 1.0))
        if self._input_color_order not in {"RGB", "BGR"}:
            raise ValueError("model.input_color_order must be RGB or BGR")
        if self._float_input_scale <= 0 or self._integer_input_scale <= 0:
            raise ValueError("model input scales must be positive")
        self.confidence_threshold = float(config.get("confidence_threshold", 0.5))
        self.class_confidence_thresholds = {
            str(label).lower(): float(threshold)
            for label, threshold in config.get(
                "class_confidence_thresholds", {}
            ).items()
        }
        self.roi = tuple(float(v) for v in config.get("roi", [0, 0, 1, 1]))
        if len(self.roi) != 4 or not (
            0 <= self.roi[0] < self.roi[2] <= 1
            and 0 <= self.roi[1] < self.roi[3] <= 1
        ):
            raise ValueError("roi must be normalized [x1,y1,x2,y2] within the frame")
        self.max_objects = int(config.get("max_objects", 1))
        self.debounce_frames = int(config.get("debounce_frames", 3))
        if self.max_objects < 0 or self.debounce_frames < 1:
            raise ValueError("max_objects must be >= 0 and debounce_frames must be >= 1")
        self.empty_belt_is_warning = bool(config.get("empty_belt_is_warning", False))
        self.size_calibration = config.get("size_calibration", {})
        self.size_calibrated = bool(self.size_calibration.get("calibrated", False))
        if not self.size_calibrated:
            LOGGER.warning(
                "Bottle-size calibration is not complete; detected bottles will not "
                "be classified as undersized or oversized until calibrated."
            )
        self.output_dir = Path(str(config.get("output_dir", "output")))
        self._candidate_event: str | None = None
        self._candidate_count = 0
        self._last_warning_event: str | None = None
        self._frame_number = 0
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if any(not 0 <= threshold <= 1
               for threshold in self.class_confidence_thresholds.values()):
            raise ValueError("class confidence thresholds must be between 0 and 1")

    @staticmethod
    def _dequantize(tensor: np.ndarray, detail: Mapping[str, Any]) -> np.ndarray:
        if tensor.dtype.kind not in {"i", "u"}:
            return tensor.astype(np.float32)
        scales = detail.get("quantization_parameters", {}).get("scales", [])
        zero_points = detail.get("quantization_parameters", {}).get("zero_points", [])
        if len(scales) == 0:
            scale, zero_point = detail.get("quantization", (0.0, 0))
            if scale == 0:
                return tensor.astype(np.float32)
            return (tensor.astype(np.float32) - float(zero_point)) * float(scale)
        if len(scales) == 1:
            return (tensor.astype(np.float32) - float(zero_points[0])) * float(scales[0])
        axis = int(detail.get("quantization_parameters", {}).get("quantized_dimension", 0))
        shape = [1] * tensor.ndim
        shape[axis] = len(scales)
        return (
            tensor.astype(np.float32) - np.asarray(zero_points).reshape(shape)
        ) * np.asarray(scales).reshape(shape)

    def _prepare_input(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        if self.config_input_order == "RGB":
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = resized.astype(np.float32)
        dtype = self.input_detail["dtype"]
        if np.issubdtype(dtype, np.integer):
            scale, zero_point = self.input_detail.get("quantization", (0.0, 0))
            if scale <= 0:
                raise ValueError("Integer model input has no valid quantization scale")
            tensor = np.rint(
                tensor * self._integer_input_scale / float(scale) + float(zero_point)
            )
            info = np.iinfo(dtype)
            tensor = np.clip(tensor, info.min, info.max).astype(dtype)
        else:
            tensor *= float(self.input_scale)
            tensor = tensor.astype(dtype)
        return np.expand_dims(tensor, axis=0)

    def _roi_crop(self, frame: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Crop the configured normalized ROI and return crop origin in pixels."""
        frame_height, frame_width = frame.shape[:2]
        x0 = int(self.roi[0] * frame_width)
        y0 = int(self.roi[1] * frame_height)
        x1 = min(frame_width, max(x0 + 1, math.ceil(self.roi[2] * frame_width)))
        y1 = min(frame_height, max(y0 + 1, math.ceil(self.roi[3] * frame_height)))
        return frame[y0:y1, x0:x1], x0, y0

    @property
    def config_input_order(self) -> str:
        return self._input_color_order

    @property
    def input_scale(self) -> float:
        return self._float_input_scale

    def predict(self, frame: np.ndarray) -> tuple[list[dict[str, Any]], float]:
        """Run the model and return detections in pixel xyxy coordinates."""
        roi_frame, roi_x, roi_y = self._roi_crop(frame)
        start = time.perf_counter()
        self.interpreter.set_tensor(
            self.input_detail["index"], self._prepare_input(roi_frame)
        )
        self.interpreter.invoke()
        inference_ms = (time.perf_counter() - start) * 1000.0

        outputs: dict[str, np.ndarray] = {}
        for name, detail in zip(self.output_order, self.output_details[:4]):
            raw = self.interpreter.get_tensor(detail["index"])
            outputs[name] = self._dequantize(raw, detail)

        boxes = np.asarray(outputs["boxes"]).reshape(-1, 4)
        classes = np.asarray(outputs["classes"]).reshape(-1)
        scores = np.asarray(outputs["scores"]).reshape(-1)
        raw_count = np.asarray(outputs["count"]).reshape(-1)
        count = int(raw_count[0]) if raw_count.size else min(len(boxes), len(scores))
        count = max(0, min(count, len(boxes), len(classes), len(scores)))

        frame_height, frame_width = frame.shape[:2]
        roi_height, roi_width = roi_frame.shape[:2]
        detections: list[dict[str, Any]] = []
        for index in range(count):
            score = float(scores[index])
            if not math.isfinite(score):
                continue
            class_id = int(round(float(classes[index])))
            label = self.labels.get(class_id, f"class_{class_id}")
            label_threshold = self.class_confidence_thresholds.get(
                label.lower(), self.confidence_threshold
            )
            if score < label_threshold:
                continue
            y1, x1, y2, x2 = (float(value) for value in boxes[index])
            x1 = min(max(roi_x + x1 * roi_width, 0.0), float(frame_width))
            x2 = min(max(roi_x + x2 * roi_width, 0.0), float(frame_width))
            y1 = min(max(roi_y + y1 * roi_height, 0.0), float(frame_height))
            y2 = min(max(roi_y + y2 * roi_height, 0.0), float(frame_height))
            box = [x1, y1, x2, y2]
            center_x = ((x1 + x2) / 2.0) / frame_width
            center_y = ((y1 + y2) / 2.0) / frame_height
            if not (
                self.roi[0] <= center_x <= self.roi[2]
                and self.roi[1] <= center_y <= self.roi[3]
            ):
                continue
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            area_ratio = (width * height) / (frame_width * frame_height)
            detection: dict[str, Any] = {
                "label": label,
                "score": score,
                "box": box,
                "area_ratio": area_ratio,
                "width_px": width,
                "height_px": height,
                "height_ratio": height / frame_height,
            }
            if label.lower() == "bottle" and self.size_calibrated:
                detection["size_event"] = classify_bottle_size(
                    box,
                    frame_width,
                    frame_height,
                    float(self.size_calibration.get("max_area_ratio", 1.0)),
                    float(self.size_calibration.get("max_height_ratio", 1.0)),
                    float(self.size_calibration.get("min_area_ratio", 0.0)),
                    float(self.size_calibration.get("min_height_ratio", 0.0)),
                )
            elif label.lower() == "bottle":
                detection["size_event"] = "NORMAL"
            detections.append(detection)
        return detections, inference_ms

    def _classify_frame(self, detections: Sequence[Mapping[str, Any]]) -> str:
        if not detections:
            return "EMPTY_BELT"
        if any(str(item["label"]).lower() != "bottle" for item in detections):
            return "FOREIGN_OBJECT"
        if any(item.get("size_event") == "OVERSIZED_BOTTLE" for item in detections):
            return "OVERSIZED_BOTTLE"
        if any(item.get("size_event") == "UNDERSIZED_BOTTLE" for item in detections):
            return "UNDERSIZED_BOTTLE"
        if len(detections) > self.max_objects:
            return "MULTIPLE_OBJECTS"
        return "NORMAL"

    def _debounced_event(self, candidate: str) -> str:
        should_warn = candidate in WARNING_EVENTS or (
            candidate == "EMPTY_BELT" and self.empty_belt_is_warning
        )
        if not should_warn:
            self._candidate_event = None
            self._candidate_count = 0
            self._last_warning_event = None
            return candidate
        if candidate == self._candidate_event:
            self._candidate_count += 1
        else:
            self._candidate_event = candidate
            self._candidate_count = 1
        if self._candidate_count < self.debounce_frames:
            return "NORMAL"
        return candidate

    @staticmethod
    def _draw(frame: np.ndarray, detections: Sequence[Mapping[str, Any]],
              status: str, event: str) -> np.ndarray:
        annotated = frame.copy()
        color = (0, 0, 255) if status == "WARNING" else (0, 180, 0)
        for detection in detections:
            x1, y1, x2, y2 = (int(round(float(v))) for v in detection["box"])
            label = f'{detection["label"]} {float(detection["score"]):.2f}'
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated, label, (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )
        cv2.putText(
            annotated, f"{status}: {event}", (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA,
        )
        return annotated

    def annotate_frame(self, frame: np.ndarray, result: Mapping[str, Any]) -> np.ndarray:
        """Return a BGR copy with the result boxes, status, and event overlaid."""
        return self._draw(
            frame, result["detections"], str(result["status"]), str(result["event"])
        )

    def analyze_frame(self, frame: np.ndarray) -> dict[str, Any]:
        """Analyze one BGR frame and return a JSON-friendly result dictionary."""
        if (frame is None or frame.ndim != 3 or frame.shape[2] != 3
                or frame.shape[0] <= 0 or frame.shape[1] <= 0):
            raise ValueError("frame must be a non-empty three-channel image")
        detections, inference_ms = self.predict(frame)
        candidate = self._classify_frame(detections)
        event = self._debounced_event(candidate)
        status = "WARNING" if event in WARNING_EVENTS or (
            event == "EMPTY_BELT" and self.empty_belt_is_warning
        ) else "NORMAL"
        self._frame_number += 1
        result: dict[str, Any] = {
            "status": status,
            "event": event,
            "detections": [
                {key: detection[key] for key in ("label", "score", "box", "area_ratio")}
                for detection in detections
            ],
            "inference_ms": inference_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "snapshot_path": None,
        }
        if status == "WARNING" and event != self._last_warning_event:
            snapshots = self.output_dir / "snapshots"
            snapshots.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            snapshot_path = snapshots / f"{stamp}_{event.lower()}.jpg"
            annotated = self.annotate_frame(frame, result)
            if not cv2.imwrite(str(snapshot_path), annotated):
                raise OSError(f"Could not write warning snapshot: {snapshot_path}")
            result["snapshot_path"] = str(snapshot_path)
        self._last_warning_event = event if status == "WARNING" else None
        return result

    def calibration_measurements(self, frame: np.ndarray) -> list[tuple[float, float]]:
        """Return (area_ratio, height_ratio) for bottle detections only."""
        detections, _ = self.predict(frame)
        frame_height, _ = frame.shape[:2]
        return [
            (float(item["area_ratio"]), float(item["height_px"]) / frame_height)
            for item in detections
            if str(item["label"]).lower() == "bottle"
        ]

def read_ground_truth(path: str | Path) -> list[tuple[float, float, str]]:
    """Read manual event intervals from start_s,end_s,expected_event CSV."""
    intervals: list[tuple[float, float, str]] = []
    with Path(path).open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"start_s", "end_s", "expected_event"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(
                "Ground truth CSV must contain start_s,end_s,expected_event columns"
            )
        for row in reader:
            start_s, end_s = float(row["start_s"]), float(row["end_s"])
            event = row["expected_event"].strip().upper()
            if start_s < 0 or end_s < start_s or not event:
                raise ValueError(f"Invalid ground-truth interval: {row}")
            intervals.append((start_s, end_s, event))
    return intervals


def evaluate_events(
    results: Sequence[Mapping[str, Any]],
    intervals: Sequence[tuple[float, float, str]],
) -> dict[str, dict[str, float | int]]:
    """Compute per-event precision and recall over analyzed frames."""
    events = set(WARNING_EVENTS)
    events.update(event for _, _, event in intervals)
    report: dict[str, dict[str, float | int]] = {}
    for event in sorted(events):
        tp = fp = fn = 0
        for result in results:
            timestamp = result.get("video_time_s")
            if timestamp is None:
                continue
            truth = any(start <= float(timestamp) <= end and expected == event
                        for start, end, expected in intervals)
            predicted = result.get("status") == "WARNING" and result.get("event") == event
            tp += int(truth and predicted)
            fp += int(not truth and predicted)
            fn += int(truth and not predicted)
        report[event] = {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
        }
    return report
