#!/usr/bin/env python3
"""
Utility script to compute DOPE cuboid dimensions from OBJ meshes.

The CLI mirrors train.py in that you can supply a JSON config with defaults
and override specific options on the command line.

Example:
    python train/compute_bbox_dimensions.py --object Antenna
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

DEFAULT_MODELS_ROOT = Path("data") / "models"
DEFAULT_OBJECT_PROPERTIES = Path("config") / "object_properties.json"


class ObjMesh:
    """Minimal OBJ parser that keeps track of vertices referenced by each group."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"OBJ file '{self.path}' does not exist.")
        self.vertices: List[Optional[Tuple[float, float, float]]] = [None]
        self.part_vertices: Dict[str, Set[int]] = {}
        self.default_part = "__default__"
        self._parse()

    def _parse(self) -> None:
        active_groups: List[str] = [self.default_part]
        total_vertices = 0
        with self.path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                prefix = line.split(maxsplit=1)[0]
                if prefix == "v":
                    tokens = line.split()
                    if len(tokens) < 4:
                        continue
                    try:
                        x, y, z = map(float, tokens[1:4])
                    except ValueError:
                        continue
                    self.vertices.append((x, y, z))
                    total_vertices += 1
                elif prefix in {"g", "o"}:
                    names = line.split()[1:]
                    if not names:
                        active_groups = [self.default_part]
                    else:
                        active_groups = names
                        for name in active_groups:
                            self.part_vertices.setdefault(name, set())
                elif prefix == "f":
                    vertex_indices = self._parse_face_indices(line)
                    if not vertex_indices:
                        continue
                    targets = active_groups or [self.default_part]
                    for group in targets:
                        pts = self.part_vertices.setdefault(group, set())
                        pts.update(vertex_indices)
                else:
                    continue
        # Clean up dangling vertices that never appear in faces.
        used_indices = set().union(*self.part_vertices.values()) if self.part_vertices else set()
        if not used_indices and total_vertices:
            # Use every vertex when no faces are assigned to groups.
            used_indices = set(range(1, total_vertices + 1))
            self.part_vertices[self.default_part] = used_indices

    def _parse_face_indices(self, line: str) -> List[int]:
        tokens = line.split()[1:]
        if not tokens:
            return []
        indices: List[int] = []
        total_vertices = len(self.vertices) - 1
        for token in tokens:
            vertex = token.split("/")[0]
            if not vertex:
                continue
            try:
                idx = int(vertex)
            except ValueError:
                continue
            if idx < 0:
                idx = total_vertices + idx + 1
            if idx < 1 or idx > total_vertices:
                continue
            indices.append(idx)
        return indices

    def available_parts(self) -> Sequence[str]:
        return sorted(name for name in self.part_vertices if self.part_vertices[name])

    def collect_vertices(self, part_names: Iterable[str]) -> np.ndarray:
        all_indices: Set[int] = set()
        for part in part_names:
            all_indices.update(self.part_vertices.get(part, set()))
        if not all_indices:
            all_indices = set(range(1, len(self.vertices)))
        coords = np.array([self.vertices[i] for i in sorted(all_indices)], dtype=float)
        return coords


def load_object_properties(config_path: Path) -> Dict[str, Dict]:
    if not config_path.is_file():
        return {}
    try:
        data = json.loads(config_path.read_text())
    except Exception as exc:
        raise SystemExit(f"[object_properties] Failed to read '{config_path}': {exc}")
    normalized = {}
    for key, value in data.items():
        if isinstance(value, dict):
            normalized[key.lower()] = value
    return normalized


def match_parts(available: Sequence[str], target: Optional[str]) -> List[str]:
    if not available:
        return []
    if not target:
        return list(available)
    needle = target.strip().lower()
    return [name for name in available if needle in name.lower()]


def parts_for_bbox(available: Sequence[str], properties: Optional[Dict], class_name: str) -> Tuple[List[str], bool]:
    include = list(available)
    if not properties:
        return include, False
    exclude_cfg = properties.get("exclude_parts")
    if not exclude_cfg:
        return include, False
    include_set: Set[str] = set(include)
    custom = False
    for entry in exclude_cfg:
        target = entry.get("name")
        matched = match_parts(available, target)
        if not matched:
            continue
        in_bbox = entry.get("in_bbox")
        if in_bbox is None:
            continue
        if bool(in_bbox):
            include_set.update(matched)
        else:
            custom = True
            include_set.difference_update(matched)
    if not custom:
        return include, False
    if not include_set:
        print(f"[warning] All parts removed from bounding box for '{class_name}'. Using full mesh.")
        return include, False
    return sorted(include_set), True


def compute_bounds(vertices: np.ndarray, unit_scale: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_vals = vertices.min(axis=0)
    max_vals = vertices.max(axis=0)
    dims = max_vals - min_vals
    if not math.isclose(unit_scale, 1.0):
        min_vals = min_vals * unit_scale
        max_vals = max_vals * unit_scale
        dims = dims * unit_scale
    return min_vals, max_vals, dims


def resolve_mesh_path(models_root: Path, object_name: str, explicit_path: Optional[Path]) -> Path:
    if explicit_path:
        return explicit_path
    obj_lower = object_name.lower()
    root = models_root
    if not root.exists():
        raise FileNotFoundError(f"Models root '{root}' does not exist.")
    if (root / f"{object_name}.obj").is_file():
        return root / f"{object_name}.obj"
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() == obj_lower:
            exact = child / f"{child.name}.obj"
            if exact.is_file():
                return exact
            for candidate in child.glob("*.obj"):
                return candidate
    for candidate in root.rglob("*.obj"):
        if obj_lower in candidate.name.lower():
            return candidate
    raise FileNotFoundError(f"Unable to find OBJ for '{object_name}' under '{models_root}'.")


def parse_args(argv: Optional[Sequence[str]] = None):
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("-c", "--config", metavar="FILE", help="JSON config with default arguments.")
    args, remaining = conf_parser.parse_known_args(argv)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--object", nargs="+", help="Object class names (match training JSON entries).")
    parser.add_argument("--models_root", type=Path, default=DEFAULT_MODELS_ROOT, help="Root directory containing OBJ assets.")
    parser.add_argument("--model_path", type=Path, default=None, help="Optional direct path to a single OBJ file.")
    parser.add_argument("--unit_scale", type=float, default=1.0, help="Scale factor applied to raw OBJ units to convert to centimeters.")
    parser.add_argument("--object_properties", type=Path, default=DEFAULT_OBJECT_PROPERTIES, help="Path to object_properties.json.")
    parser.add_argument("--show_parts", action="store_true", help="Print the mesh parts that contribute to the bounding box.")

    config_path: Optional[Path] = None
    if args.config:
        config_path = Path(args.config).expanduser()
    else:
        default_cfg = Path(__file__).resolve().parents[1] / "config" / "training_config.json"
        if default_cfg.exists():
            config_path = default_cfg
    if config_path:
        try:
            raw_config = json.loads(config_path.read_text())
        except Exception as exc:
            raise SystemExit(f"[config] Failed to read JSON config '{config_path}': {exc}")
        valid_dests = {
            action.dest
            for action in parser._actions
            if hasattr(action, "dest") and action.dest not in (argparse.SUPPRESS, "help")
        }
        defaults = {k: v for k, v in raw_config.items() if k in valid_dests}
        if defaults:
            parser.set_defaults(**defaults)
    opts = parser.parse_args(remaining)
    if not opts.object:
        raise SystemExit("--object must be specified via CLI or config.")
    return opts


def main(argv: Optional[Sequence[str]] = None) -> int:
    opt = parse_args(argv)
    properties = load_object_properties(opt.object_properties)
    results = []
    for obj_name in opt.object:
        mesh_path = resolve_mesh_path(opt.models_root, obj_name, opt.model_path)
        mesh = ObjMesh(mesh_path)
        available_parts = mesh.available_parts()
        obj_props = properties.get(obj_name.lower())
        selected_parts, custom = parts_for_bbox(available_parts, obj_props, obj_name)
        vertices = mesh.collect_vertices(selected_parts)
        min_vals, max_vals, dims = compute_bounds(vertices, opt.unit_scale)
        results.append((obj_name, mesh_path, selected_parts, dims))
        print(f"\nObject: {obj_name}")
        print(f"  Mesh: {mesh_path}")
        if opt.show_parts:
            excluded = sorted(set(available_parts) - set(selected_parts))
            print(f"  Included parts: {', '.join(selected_parts) if selected_parts else '(all)'}")
            if custom:
                print(f"  Excluded parts: {', '.join(excluded) if excluded else '(none)'}")
        print(f"  Dimensions (cm): [{dims[0]:.6f}, {dims[1]:.6f}, {dims[2]:.6f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
