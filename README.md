# Data generation
```bash
# in project root
python3 ./data_generation/blenderproc_data_gen/run_blenderproc_datagen.py --nb_runs 1 --frames_per_cycle 1 --path_single_obj ./data/models/antenna/antenna.obj --nb_objects 1 --object_class Antenna --distractors_folder ./data/google_scanned_models/ --nb_distractors 5 --backgrounds_folder ./data/dome_hdri_haven/ --outf ./data/AntennaData
```
Optional: limit object rotations by creating `config/sence_config.json` (or pass `--rotation_config <path>`). Wrap the ranges inside the `rotational_range` key:
```json
{
  "rotational_range": {
    "default": {"pitch": [-10, 10], "yaw": [-180, 180], "roll": [-10, 10]},
    "ketchup": {"pitch": [-5, 5], "yaw": [0, 360], "roll": [-5, 5]}
  }
}
```
Values are degrees and can be defined per class name (case-insensitive) plus an optional `default`.

HDRI backgrounds can also be managed from the same file via an `HDRI_backgrounds` block:
```json
{
  "HDRI_backgrounds": {
    "enabled": true,
    "selection_method": "iterate",
    "strength_range": [0.5, 1.5],
    "count": "all"
  }
}
```
When enabled, only HDR (`.hdr`/`.exr`) images from `--backgrounds_folder` are used. `selection_method: "random"` samples `count` unique HDRIs (without replacement) per run—set `count` to a number or `"all"` (default) to use every HDR. `selection_method: "iterate"` also cycles through all HDRs deterministically. `strength_range` specifies the min/max environment strength sampled per frame. Total frames contributed by HDR cycling equal `frames_per_cycle * (# of HDRs used in the run)`.

Foreground/background compositing can be controlled via the `backgrounds` block (images default to `./data/background_images` when `folder` is omitted):
```json
{
  "backgrounds": {
    "enabled": true,
    "selection_method": "iterate",
    "count": 3,
    "folder": "./data/background_images"
  }
}
```
`selection_method: "random"` samples `count` unique backgrounds (without replacement) for each HDR in the run; omit `count` (or set `"all"`) to draw from every image. `iterate` (or `"all"`) also pairs each HDR with `count` distinct backgrounds but does so deterministically. Total frames become `frames_per_cycle * (# of HDRs when iterating) * count`.

Optional emissive/object properties can be defined via `config/object_properties.json` (or `--object_properties`). Example:
```json
{
  "antenna": {
    "emissive": [
      {
        "body": "LED",
        "color": "random",
        "color_list": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "intensity": 15
      }
    ]
  }
}
```
During generation the mesh whose name contains `LED` will get an emissive material with a randomly chosen color from the list.

# Training Data
```bash
# in project root
torchrun --nproc_per_node=1 train/train.py --data  ./data/KetchupData --object Ketchup --epochs 5 --save_every 5
```

# Validate Data
```bash
python3 ./data_generation/validate_data.py ./data/AntennaData/0/*.json
```

# Generate Test data
```bash
# in project root
python3 ./data_generation/blenderproc_data_gen/run_blenderproc_datagen.py --nb_runs 8 --frames_per_cycle 1000 --path_single_obj ./data/models/antenna/antenna.obj --nb_objects 1 --object_class Antenna --distractors_folder ./data/google_scanned_models/ --nb_distractors 5 --backgrounds_folder ./data/dome_hdri_haven/ --outf ./data/AntennaData
```

# Image Inference
```bash
# in project root
python3 ./inference/inference.py --config_dir ./config --object Ketchup --parallel --weights ./weights/Ketchup.pth --data ./data/KetchupTest/
```

# Camera Inference
```bash
# in project root
python3 ./inference/inference.py \
  --config_dir ./config \
  --object Ketchup \
  --weights ./weights/Ketchup.pth \
  --webcam 0 \
  --webcam_display
```

# Video Inference
```bash
# in project root
python3 ./inference/inference.py --config_dir ./config --object Ketchup --weights ./weights/Ketchup.pth --video ./data/KetchupTest/videos/video.mp4 --silence-detection --video_out ./output/video_inference.mp4
```
