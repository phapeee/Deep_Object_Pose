"""
things needed 

predictions for image 
ground thruth for that image 
3d model loaded 

compare the poses.

"""


import argparse
import csv
import math
import os
import sys
try:
    from dataclasses import dataclass
except ImportError:  # Python < 3.7 fallback
    def dataclass(cls):
        return cls
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
try:
    import simplejson as json
except ImportError:  # fall back to stdlib when simplejson unavailable
    import json
try:
    from scipy import spatial
except ImportError:
    class _SpatialModule:
        @staticmethod
        def distance_matrix(x, y, p=2):
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
                raise ValueError("distance_matrix inputs must be 2D arrays with matching feature dimensions")
            diff = x[:, None, :] - y[None, :, :]
            if p == 2:
                return np.sqrt(np.sum(diff * diff, axis=-1))
            diff = np.sum(np.abs(diff) ** p, axis=-1)
            return diff ** (1.0 / p)
    spatial = _SpatialModule()

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _log(message):
    """Print evaluation progress messages with timestamps."""
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[evaluate {now}] {message}")
    sys.stdout.flush()


def progress_iterable(iterable, total=None, desc=""):
    if tqdm is None:
        if desc:
            _log(desc)
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def load_groundtruth(root: str) -> List[str]:
    root_path = Path(root).resolve()
    gts: List[str] = []

    directories: List[Path] = []

    def explore(path: Path):
        if not path.is_dir():
            return
        directories.append(path)
        for child in sorted(path.iterdir()):
            if child.is_dir():
                explore(child)

    explore(root_path)

    iterator = progress_iterable(
        directories,
        total=len(directories),
        desc="Scanning GT folders",
    )
    skip_files = {".dope_processed_folders.json", "training_manifest.json"}

    for path in iterator:
        relative = [
            p
            for p in path.iterdir()
            if p.is_file()
            and p.suffix == ".json"
            and "settings" not in p.name
            and p.name not in skip_files
        ]
        for json_path in relative:
            rel = json_path.resolve().relative_to(root_path).as_posix()
            gts.append(rel)

    return gts


def load_prediction(root: str, groundtruths: Sequence[str]) -> List[str]:
    """
    Supports multiple prediction folders for one set of testing data.
    Each prediction folder must contain the same folder structure as the testing data directory.
    """
    root_path = Path(root).resolve()
    subdirs = [d.name for d in root_path.iterdir() if d.is_dir()]
    subdirs.append("")  # also allow predictions stored directly in root

    prediction_folders: List[str] = []
    for directory in sorted(subdirs):
        valid_folder = True
        for gt in groundtruths:
            file_path = root_path / directory / gt
            if not file_path.exists():
                valid_folder = False
                break
        if valid_folder:
            prediction_folders.append(directory)
    return prediction_folders


def calculate_auc(thresholds, add_list, total_objects):
    res = []
    add_array = np.asarray(add_list, dtype=float)
    for thresh in thresholds:
        under_thresh = len(np.where(add_array <= thresh)[0]) / total_objects
        res.append(under_thresh)
    return res


def calculate_auc_total(add_list, total_objects, delta_threshold=0.00001, max_threshold=0.1):
    add_array = np.asarray(add_list, dtype=float)
    add_threshold_values = np.arange(0.0, max_threshold, delta_threshold)
    counts = []
    for value in add_threshold_values:
        under_threshold = len(np.where(add_array <= value)[0]) / total_objects
        counts.append(under_threshold)
    auc = np.trapz(counts, dx=delta_threshold) / max_threshold
    return auc


def _load_obj_vertices(obj_path: Path) -> np.ndarray:
    vertices: List[Tuple[float, float, float]] = []
    with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            if not raw_line or raw_line[0] not in {"v", "V"}:
                continue
            line = raw_line.strip()
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            except ValueError:
                continue
    return np.asarray(vertices, dtype=float)


def _cuboid_points_from_vertices(vertices: np.ndarray) -> np.ndarray:
    if vertices.size == 0:
        return np.zeros((9, 3), dtype=float)
    min_vals = vertices.min(axis=0)
    max_vals = vertices.max(axis=0)
    cx = (min_vals[0] + max_vals[0]) / 2.0
    cy = (min_vals[1] + max_vals[1]) / 2.0
    cz = (min_vals[2] + max_vals[2]) / 2.0
    cuboid = [
        [max_vals[0], min_vals[1], max_vals[2]],
        [max_vals[0], max_vals[1], max_vals[2]],
        [max_vals[0], max_vals[1], min_vals[2]],
        [max_vals[0], min_vals[1], min_vals[2]],
        [min_vals[0], min_vals[1], max_vals[2]],
        [min_vals[0], max_vals[1], max_vals[2]],
        [min_vals[0], max_vals[1], min_vals[2]],
        [min_vals[0], min_vals[1], min_vals[2]],
        [cx, cy, cz],
    ]
    return np.asarray(cuboid, dtype=float)


@dataclass
class MeshData:
    name: str
    vertices: np.ndarray
    cuboid: np.ndarray


class MeshLibrary:
    def __init__(self, root: str, scale: float = 0.01):
        self.scale = scale
        self.meshes: Dict[str, MeshData] = {}
        root_path = Path(root)
        if root_path.is_file() and root_path.suffix.lower() == ".obj":
            self._add_model(root_path)
        else:
            self._scan_directory(root_path)
        if not self.meshes:
            raise SystemExit(f"No OBJ files found under '{root}'. Cannot evaluate.")

    def _add_model(self, obj_path: Path):
        vertices = _load_obj_vertices(obj_path)
        if vertices.size == 0:
            return
        scaled = vertices * self.scale
        cuboid = _cuboid_points_from_vertices(scaled)
        name = obj_path.parent.name or obj_path.stem
        key = name.lower()
        if key in self.meshes:
            _log(f"Warning: duplicate mesh name '{name}' encountered; keeping the first instance.")
            return
        self.meshes[key] = MeshData(name=name, vertices=scaled, cuboid=cuboid)

    def _scan_directory(self, root: Path):
        if not root.exists():
            raise SystemExit(f"Model path '{root}' does not exist.")
        for dirpath, _, filenames in os.walk(root):
            dir_path = Path(dirpath)
            textured = dir_path / "textured_simple.obj"
            if textured.exists():
                self._add_model(textured)
                continue
            for filename in filenames:
                if filename.lower().endswith(".obj"):
                    self._add_model(dir_path / filename)

    def get(self, name: str) -> Optional[MeshData]:
        if not name:
            return None
        return self.meshes.get(name.lower())


def quaternion_from_xyzw(values: Sequence[float]) -> np.ndarray:
    if len(values) < 4:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    wxyz = np.array(
        [values[3], values[0], values[1], values[2]],
        dtype=float,
    )
    norm = np.linalg.norm(wxyz)
    if norm == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return wxyz / norm


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quaternion_from_axis_angle(angle: float, axis: Sequence[float]) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis = axis / axis_norm
    half = angle / 2.0
    sin_half = math.sin(half)
    return np.array(
        [math.cos(half), axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half],
        dtype=float,
    )


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def transform_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    return (rotation @ points.T).T + translation


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_prediction",
        default="../inference/output",
        help="Path to prediction data.",
    )
    parser.add_argument(
        "--data",
        default="../sample_data/",
        help="path to ground truth data.",
    )
    parser.add_argument(
        "--models",
        default="../3d_models/YCB_models/",
        help="Path to the 3D object models (folder or .obj file).",
    )
    parser.add_argument("--outf", default="output", help="where to put the results.")
    parser.add_argument(
        "--adds", action="store_true", help="run ADDS, this might take a while"
    )
    parser.add_argument(
        "--cuboid", action="store_true", help="use cuboid to compute the ADD"
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        default=[0.02, 0.04, 0.06, 0.08, 0.10],
        help="Thresholds to compute accuracy for.",
    )
    parser.add_argument(
        "--model_scale",
        type=float,
        default=0.01,
        help="Scaling applied to OBJ vertices (default assumes OBJ units are centimeters).",
    )
    parser.add_argument(
        "--location_scale",
        type=float,
        default=0.01,
        help="Scaling applied to translation vectors from JSON files (default converts centimeters to meters).",
    )

    opt = parser.parse_args()

    _log(
        "Starting evaluation with configuration:\n"
        f"  predictions={opt.data_prediction}\n"
        f"  ground_truth={opt.data}\n"
        f"  models={opt.models}\n"
        f"  outf={opt.outf}\n"
        f"  adds={opt.adds}\n"
        f"  cuboid={opt.cuboid}\n"
        f"  thresholds={opt.thresholds}\n"
        f"  model_scale={opt.model_scale}\n"
        f"  location_scale={opt.location_scale}"
    )

    mesh_library = MeshLibrary(opt.models, scale=opt.model_scale)
    _log(f"Loaded {len(mesh_library.meshes)} mesh definition(s).")

    # Get Ground Truth and Prediction Data
    _log("Reading ground-truth annotations.")
    data_truth = load_groundtruth(opt.data)
    _log(f"Discovered {len(data_truth)} annotation files.")
    _log("Collecting prediction folders.")
    prediction_folders = load_prediction(opt.data_prediction, data_truth)
    prediction_folders.sort()

    print(f"Number of Ground Truth Data : {len(data_truth)}")
    print(f"Number of Prediction Folders: {len(prediction_folders)}")
    _log("Beginning evaluation runs.")

    # Make output directory if it does not exist
    os.makedirs(os.path.join(opt.outf), exist_ok=True)
    csv_path = os.path.join(opt.outf, "result.csv")
    _log(f"Writing evaluation summary to {csv_path}")

    csv_file = open(csv_path, "w+")

    csv_writer = csv.writer(csv_file)

    # Write Header Row
    csv_writer.writerow(["--cuboid", opt.cuboid, "--adds", opt.adds])
    csv_writer.writerow(["Weights", "Object", "Total AUC"] + opt.thresholds)

    missing_models: set = set()

    iterable_folders = progress_iterable(
        prediction_folders,
        total=len(prediction_folders),
        desc="Prediction Folders",
    )
    for pf_i, pf in enumerate(iterable_folders, start=1):
        _log(f"[{pf_i}/{len(prediction_folders)}] Evaluating predictions in '{pf}'.")
        adds_objects = {}
        adds_all = []

        count_all_gt = 0
        count_by_object = {}

        count_all_preds = 0
        count_by_object_preds = {}

        iterable_gt = progress_iterable(
            data_truth,
            total=len(data_truth),
            desc=f"GT Files ({pf or 'root'})",
        )
        for f_i, gt_file in enumerate(iterable_gt, start=1):

            gt_path = os.path.join(opt.data, gt_file)
            pred_path = os.path.join(opt.data_prediction, pf, gt_file)

            with open(gt_path) as json_file:
                gt_json = json.load(json_file)

            with open(pred_path) as json_file:
                gu_json = json.load(json_file)

            objects_gt = []
            for obj in gt_json.get("objects", []):
                name_gt = obj.get("class", "")
                name_gt = name_gt[:-4] if name_gt.endswith("_16k") else name_gt
                canonical_name = name_gt.lower()
                rotation = quaternion_to_matrix(quaternion_from_xyzw(obj.get("quaternion_xyzw", [0, 0, 0, 1])))
                location_values = obj.get("location", [0.0, 0.0, 0.0])
                position = np.asarray(location_values, dtype=float) * opt.location_scale
                objects_gt.append(
                    {
                        "name": canonical_name,
                        "display_name": name_gt,
                        "rotation": rotation,
                        "position": position,
                    }
                )
                count_all_gt += 1
                count_by_object[canonical_name] = count_by_object.get(canonical_name, 0) + 1

            for obj_pred in gu_json.get("objects", []):
                name_pred = obj_pred.get("class", "")
                name_pred = name_pred[:-4] if name_pred.endswith("_16k") else name_pred
                canonical_pred = name_pred.lower()

                quat_pred = quaternion_from_xyzw(obj_pred.get("quaternion_xyzw", [0, 0, 0, 1]))
                quat_pred = quaternion_multiply(quat_pred, quaternion_from_axis_angle(1.57, (1, 0, 0)))
                quat_pred = quaternion_multiply(quat_pred, quaternion_from_axis_angle(1.57, (0, 0, 1)))
                rot_pred = quaternion_to_matrix(quat_pred)

                try:
                    position_pred = np.asarray(obj_pred.get("location", [0.0, 0.0, 0.0]), dtype=float) * opt.location_scale
                except Exception:
                    position_pred = np.array([1e6, 1e6, 1e6], dtype=float)

                pose_mesh = {
                    "rotation": rot_pred,
                    "position": position_pred,
                }

                count_all_preds += 1
                count_by_object_preds[canonical_pred] = count_by_object_preds.get(canonical_pred, 0) + 1

                mesh_pred = mesh_library.get(canonical_pred)
                if mesh_pred is None:
                    if canonical_pred not in missing_models:
                        _log(f"Warning: no mesh found for object '{name_pred}'. Skipping its predictions.")
                        missing_models.add(canonical_pred)
                    continue

                candidates = [
                    (idx, pose_gt)
                    for idx, pose_gt in enumerate(objects_gt)
                    if pose_gt["name"] == canonical_pred
                ]

                best_dist = float("inf")
                best_index = None

                for i_gt, pose_gt in candidates:
                    mesh_gt = mesh_library.get(pose_gt["name"])
                    if mesh_gt is None:
                        if pose_gt["name"] not in missing_models:
                            _log(f"Warning: no mesh found for object '{pose_gt['display_name']}'.")
                            missing_models.add(pose_gt["name"])
                        continue

                    if opt.cuboid:
                        points_gt = transform_points(mesh_gt.cuboid, pose_gt["rotation"], pose_gt["position"])
                        points_gu = transform_points(mesh_pred.cuboid, pose_mesh["rotation"], pose_mesh["position"])
                    else:
                        points_gt = transform_points(mesh_gt.vertices, pose_gt["rotation"], pose_gt["position"])
                        points_gu = transform_points(mesh_pred.vertices, pose_mesh["rotation"], pose_mesh["position"])

                    if points_gt.size == 0 or points_gu.size == 0:
                        continue

                    if opt.adds:
                        dist_matrix = spatial.distance_matrix(points_gt, points_gu, p=2)
                        dist = float(np.mean(np.min(dist_matrix, axis=1)))
                    else:
                        min_len = min(len(points_gt), len(points_gu))
                        if min_len == 0:
                            continue
                        paired = points_gt[:min_len] - points_gu[:min_len]
                        dist = float(np.mean(np.linalg.norm(paired, axis=1)))

                    if dist < best_dist:
                        best_dist = dist
                        best_index = i_gt

                if best_index is not None:
                    adds_objects.setdefault(canonical_pred, []).append(best_dist)
                    adds_all.append(best_dist)

        # Compute Metrics
        for canonical_name, distances in adds_objects.items():
            mesh_info = mesh_library.get(canonical_name)
            display_name = mesh_info.name if mesh_info else canonical_name

            auc_thresh = calculate_auc(
                thresholds=opt.thresholds,
                add_list=np.array(distances),
                total_objects=count_by_object.get(canonical_name, len(distances)),
            )

            auc_total = calculate_auc_total(
                add_list=np.array(distances),
                total_objects=count_by_object.get(canonical_name, len(distances)),
            )

            csv_writer.writerow([pf, display_name, auc_total] + auc_thresh)
            _log(
                f"Completed object '{display_name}' in '{pf}': Total AUC={auc_total:.4f}, "
                f"Threshold AUCs={['{:.4f}'.format(v) for v in auc_thresh]}"
            )

        scored_names = [mesh_library.get(name).name if mesh_library.get(name) else name for name in adds_objects.keys()]
        _log(f"Finished evaluating '{pf}'. Objects scored: {scored_names}")

    csv_file.close()
    _log(f"Results saved to {csv_path}")

    _log("Evaluation complete.")
