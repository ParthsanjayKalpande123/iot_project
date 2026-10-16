# Conveyor-belt TFLite vision module

Offline object detection for a conveyor that should carry one 1 L bottle at a
time. The module flags oversized bottles, detected non-bottle objects, and
excess objects in the configured belt region. It uses the same
`Detector.analyze_frame()` path for video, laptop webcam, and Picamera2 input.

## Model and output assumptions

The supplied configuration and model target the COCO EfficientDet-Lite0 TFLite
detector. The default tensor order is **output tensor 0 =
boxes, 1 = classes, 2 = scores, 3 = count**. Boxes are normalized
`[ymin, xmin, ymax, xmax]`; detections are returned as pixel `[x1,y1,x2,y2]`.
Change `model.output_order` if your converted model uses another order.

Input dimensions and tensor dtype are read from the interpreter. Quantized
integer input uses the model's quantization parameters after applying
`model.integer_input_scale` (default `1`, for common 0-255 image inputs); float
input uses `model.float_input_scale` (default `1/255`). Model input color order,
labels path, and class-index offset are configurable. For a plain one-label-per-
line file, indexes start at `model.label_index_offset`; indexed files such as
`1 person` use the explicit indexes in the file.

COCO's `bottle` class does not encode volume. The calibrated upper limits for
normalized bounding-box area and height determine whether a bottle is
oversized. Camera geometry, camera distance, belt position, and bottle
orientation affect those limits; calibrate after mounting the final camera.
The calibration also stores lower bounds: bottles below both lower bounds are
reported as `UNDERSIZED_BOTTLE`; bottles within the calibrated range are
`NORMAL`. Run calibration with the same fixed camera position and belt ROI
that will be used during monitoring.
Until calibration has been completed, detected bottles are not classified by
size (to avoid arbitrary default thresholds marking normal bottles as wrong).
The runner logs a warning in this state.

## Laptop installation

Python 3.10 or newer is recommended. Use a TFLite runtime wheel compatible
with your operating system and Python version.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `ai-edge-litert` is unavailable for the laptop's Python/platform, install a
compatible `tflite-runtime` wheel instead. The program tries `ai-edge-litert`
first and then `tflite-runtime`.

## Raspberry Pi 4 installation

Install the Raspberry Pi OS camera stack and Picamera2 with the OS package
manager. Use a virtual environment that can access the OS-provided Picamera2
package, then install the project dependencies:

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-venv
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the interpreter package is not available for your 64-bit OS/Python version,
install a matching `tflite-runtime` wheel. Do not install a wheel built for a
different Python version or CPU architecture. `--show` needs a desktop display;
headless SSH operation works without that flag.

## Download model and labels

The included `models/labels.txt` uses the COCO category IDs expected by
TensorFlow's EfficientDet-Lite0 model (the detector output class index is the
COCO category ID minus one; unused category IDs are intentionally absent).
For this particular model, the configured output order is boxes, classes,
scores, count. Its uint8 input represents values in `[0,1]`, so the configured
`integer_input_scale` is `1/255`.

To download the official EfficientDet-Lite0 TFLite model, from the project
directory run:

```powershell
New-Item -ItemType Directory -Force models | Out-Null
curl.exe -L "https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/rpi/lite-model_efficientdet_lite0_detection_metadata_1.tflite" -o "models\efficientdet_lite0.tflite"
```

The URL above is the TensorFlow Lite Raspberry Pi object-detection example's
model URL. The model/license metadata is embedded in the downloaded artifact.
If switching to another model or conversion, check its output tensor order,
label indexing, input dtype and quantization, then update the configuration and
labels to match.

## Configuration and use

Clone the repository, install dependencies as above, and place a test video at
`videos/demo.mp4` (or change `video_path` in `config.yaml`). Videos are
excluded from Git because demo footage can be large or private. Then edit
`config.yaml`:

- `source`: `video` or `camera`.
- `camera.backend`: `webcam` for an OpenCV USB/integrated camera on a laptop,
  or `picamera2` for the Raspberry Pi Camera Module.
- `video_path`: demo video path; relative paths resolve from `config.yaml`.
- `roi`: normalized belt rectangle `[x1,y1,x2,y2]`.
- `confidence_threshold`, optional per-class `class_confidence_thresholds`,
  `max_objects`, and `debounce_frames`.
- `video_sample_interval_s`: process every frame at `0`; otherwise sample by
  source-video seconds.
- `output_dir`: contains `results.csv`, `output_annotated.mp4`, and warning
  JPEGs in `snapshots/`.
- `empty_belt_is_warning`: defaults to false; an empty region is still
  reported with `event=EMPTY_BELT`.

Relative video, model, and output paths resolve from the directory containing
`config.yaml`, so the sample paths remain portable after cloning.

The ROI is cropped before inference, then detection boxes are mapped back into
full-frame coordinates. For a fixed overhead camera, set it tightly around the
belt to avoid treating workers or machinery outside the belt as foreign
objects and to give belt objects more model input pixels. Bottle predictions
use a separate `0.3` threshold in the example config to improve recall; raise
it if false bottle detections appear, and tune it on labeled footage.

To test live on a laptop webcam, set `source: camera` and `camera.backend:
webcam` in `config.yaml`, then run:

```powershell
python run_demo.py --config config.yaml --show
```

Set `camera.index` to `1` or another device index if the default webcam is not
selected. Keep `backend: picamera2` for the Pi Camera Module.

Before production use, calibrate with only normal 1 L bottles visible:

```bash
python run_demo.py --config config.yaml --calibrate --calib-start 10 --calib-end 25
```

This calibration establishes the acceptable 1 L size range. It does not learn
bottle volume intrinsically; size decisions are valid only when bottles pass
through a stable camera view at a consistent distance. A 2 L bottle should be
tested in that same setup and should exceed the calibrated upper limits.

For camera calibration, set `source: camera`; the program collects 30 frames:

```bash
python run_demo.py --config config.yaml --calibrate
```

Start fast video analysis (default), or play it at original speed and/or show
a preview:

```bash
python run_demo.py --config config.yaml
python run_demo.py --config config.yaml --realtime --show
```

To evaluate against manual intervals, create `labels.csv` with the header
`start_s,end_s,expected_event`; use event names such as `OVERSIZED_BOTTLE`,
`UNDERSIZED_BOTTLE`, `FOREIGN_OBJECT`, or `MULTIPLE_OBJECTS`:

```bash
python run_demo.py --config config.yaml --ground-truth labels.csv
```

Evaluation is frame-level precision/recall at the analyzed timestamps. Warning
episodes are debounced, so the warning begins only after the configured number
of consecutive analyzed frames.

## Importing and testing

```python
from vision import Detector

detector = Detector(config)
result = detector.analyze_frame(bgr_frame)
annotated = detector.annotate_frame(bgr_frame, result)
```

`analyze_frame()` returns status, event, filtered detections, inference time,
timestamp, and a `snapshot_path` (a path when a warning episode begins,
otherwise `None`; one image is saved per episode to avoid flooding storage).
`annotate_frame()` produces the image for video/preview output.

Run the pure size-rule tests with:

```bash
python -m unittest -v test_size_rule.py
```

## GitHub contents

The repository includes the Python source, configuration, requirements, COCO
labels, and the 4.5 MB pretrained model. Local virtual environments, generated
`output/`, demo videos, and the image dataset are excluded by `.gitignore`.
The downloaded model contains Apache 2.0 license metadata.
