#!/usr/bin/env -S blenderproc run

import blenderproc as bp  # must be first!
from blenderproc.python.utility.Utility import Utility
import bpy

import argparse
import copy
import cv2
import glob
import json
from math import acos, atan, cos, pi, sin, sqrt
import numpy as np
import os
from pathlib import Path
from PIL import Image, ImageDraw
from pyquaternion import Quaternion
import random
import sys

HDR_FILE_EXTENSIONS = (".hdr", ".exr")
LDR_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def random_object_position(near=5.0, far=40.0):
    # Specialized function to randomly place the objects in a visible
    # location
    x = 20 - 40*random.random()
    y = near + (far-near)*random.random()
    z = 20 - 40*random.random()
    return np.array([x, y, z])


def random_depth_in_frustrum(tw, th, bw, bh, depth):
    '''
    Generate a random depth within a frustrum. In order to get a uniform volume
    distribution, we want the probability density function to be proportional to
    the cross-sectional area of the frustum at the generated depth.

    tw, th - the width and height, respectively at the "top" (narrowest part)
             of the frustrum
    bw, bh - the width and height, respectively at the "bottom" (widest part)
             of the frustrum
    depth  - depth of frustrum (distance between 'top' and 'bottom')
    '''
    A = (tw - bw) * (th - bw)/(depth * depth)
    B = (bw * (tw - bw) + bw * (th - bw))/depth
    C = bw * bw
    area = depth * (C + depth * (0.5 * B + depth * A / 3.0))
    r = random.random() * area
    det = B * B - 4 * A * C
    part1 = B * (B * B - 6 * A * C) - 12 * A * A * r
    part2 = sqrt(part1 * part1 - det * det * det)
    part3 = pow(part1 + part2, 1./3.)
    return (-(B + det / part3 + part3) / (2 * A))


def point_in_frustrum(camera, near=10, far=20):
    fov_w, fov_h = camera.get_fov()

    tw = sin(fov_w)*near # top (nearest to camera) width of frustrum
    th = sin(fov_h)*near # top (nearest to camera) height of frustrum
    bw = sin(fov_w)*far  # bottom width
    bh = sin(fov_h)*far  # bottom height

    # calculate random inverse depth: 0 at the 'far' plane and 1 at the 'near' plane
    inv_depth = random_depth_in_frustrum(tw, th, bw, bh, far-near)
    depth = far-inv_depth

    nd = depth/(far-near) # normalized depth
    w = nd*(bw-tw)
    h = nd*(bh-th)

    # construct points so that we are looking down -Z, +Y is up, +X to right
    x = (0.5 - random.random())*w
    y = (0.5 - random.random())*h
    z = depth

    # orient them along the camera's view direction
    xform = camera.get_camera_pose()
    return (np.array([x,y,z,1]) @ xform)[0:3]


def Rx(A):
    return np.array([[1,      0,      0],
                     [0, cos(A), -sin(A)],
                     [0, sin(A),  cos(A)]])

def Ry(A):
    return np.array([[ cos(A), 0, sin(A)],
                     [      0, 1,      0],
                     [-sin(A), 0, cos(A)]])

def Rz(A):
    return np.array([[cos(A), -sin(A), 0],
                     [sin(A),  cos(A), 0],
                     [0,            0, 1]])

def ur():
    return 2.0*random.random() - 1.0

def random_rotation_matrix(ranges=None, max_angle=180):
    """Return rotation matrix with optional per-axis ranges in degrees."""
    if ranges is not None:
        def pick_angle(axis):
            deg_range = ranges.get(axis, [-max_angle, max_angle])
            return pi * (random.uniform(deg_range[0], deg_range[1]) / 180.0)
        rx = pick_angle("pitch")
        ry = pick_angle("yaw")
        rz = pick_angle("roll")
    else:
        mr = pi*(max_angle/180.0)
        rx = mr*ur()
        ry = mr*ur()
        rz = mr*ur()
    # Orient the board so a white square (sq #0) in UL corner
    RY = Ry(-0.5*pi)
    # add some random rotations
    return RY @ Rx(rx) @ Ry(ry) @ Rz(rz)


def _default_config_path(filename):
    """Prefer the repo-wide config directory, fall back to legacy location."""
    repo_root = Path(__file__).resolve().parents[2]
    config_candidate = repo_root / "config" / filename
    if config_candidate.is_file():
        return config_candidate
    legacy_candidate = Path(__file__).with_name(filename)
    if legacy_candidate.is_file():
        return legacy_candidate
    return None


def load_sence_config_data(config_path):
    candidate = None
    if config_path:
        candidate = Path(config_path)
    else:
        candidate = _default_config_path("sence_config.json")
        if candidate is None:
            candidate = _default_config_path("rotation_config.json")
    if candidate is None:
        return None
    if not candidate.is_file():
        print(f"Sence config file '{candidate}' not found, ignoring.")
        return None
    try:
        return json.loads(candidate.read_text())
    except Exception as exc:
        print(f"Failed to read sence config '{candidate}': {exc}")
        return None


def load_rotation_config(config_path, config_data=None):
    """Load per-class rotation limits from JSON."""
    data = config_data if config_data is not None else load_sence_config_data(config_path)
    if not data:
        return None

    raw_ranges = data
    if isinstance(data, dict) and isinstance(data.get("rotational_range"), dict):
        raw_ranges = data["rotational_range"]

    normalized = {}

    def sanitize_range(values):
        if isinstance(values, (list, tuple)) and len(values) == 2:
            lo = float(values[0])
            hi = float(values[1])
            return [min(lo, hi), max(lo, hi)]
        return None

    for key, ranges in raw_ranges.items():
        if not isinstance(ranges, dict):
            continue
        entry = {}
        for axis in ("pitch", "yaw", "roll"):
            rng = sanitize_range(ranges.get(axis))
            if rng is not None:
                entry[axis] = rng
        normalized[key.lower()] = entry

    return normalized if normalized else None


def load_hdri_background_config(config_path, config_data=None):
    data = config_data if config_data is not None else load_sence_config_data(config_path)
    if not data:
        return None
    hdri_cfg = data.get("HDRI_backgrounds") or data.get("hdri_backgrounds")
    if not isinstance(hdri_cfg, dict):
        return None
    if not bool(hdri_cfg.get("enabled", False)):
        return None
    selection = str(hdri_cfg.get("selection_method", "random")).strip().lower()
    if selection not in {"random", "iterate"}:
        selection = "random"
    strength_vals = hdri_cfg.get("strength_range", [0.5, 1.5])
    low, high = 0.5, 1.5
    if isinstance(strength_vals, (list, tuple)) and len(strength_vals) >= 2:
        try:
            low = float(strength_vals[0])
            high = float(strength_vals[1])
        except (TypeError, ValueError):
            low, high = 0.5, 1.5
    if low > high:
        low, high = high, low
    raw_count = hdri_cfg.get("count")
    count_value = None
    if isinstance(raw_count, str):
        if raw_count.strip().lower() == "all":
            count_value = "all"
        else:
            try:
                count_value = int(raw_count)
            except ValueError:
                count_value = None
    elif isinstance(raw_count, (int, float)):
        count_value = int(raw_count)

    return {
        "selection_method": selection,
        "strength_range": (low, high),
        "count": count_value
    }


def load_background_overlay_config(config_path, config_data=None):
    data = config_data if config_data is not None else load_sence_config_data(config_path)
    if not data:
        return None
    bg_cfg = data.get("backgrounds") or data.get("background_images")
    if not isinstance(bg_cfg, dict):
        return None
    if not bool(bg_cfg.get("enabled", False)):
        return None
    selection = str(bg_cfg.get("selection_method", "random")).strip().lower()
    if selection == "all":
        selection = "iterate"
    if selection not in {"random", "iterate"}:
        selection = "random"
    count = bg_cfg.get("count")
    if count is not None:
        try:
            count = int(count)
        except (TypeError, ValueError):
            count = None
        if count is not None and count < 1:
            count = None
    folder_value = bg_cfg.get("folder") or bg_cfg.get("path") or bg_cfg.get("dataset")
    repo_root = Path(__file__).resolve().parents[2]
    folder_path = Path(folder_value) if folder_value else (repo_root / "data" / "background_images")
    return {
        "selection_method": selection,
        "count": count,
        "folder": folder_path
    }


def select_backgrounds_for_hdr(runtime):
    if not runtime or not runtime["paths"]:
        return [None]
    method = runtime["method"]
    paths = runtime["paths"]
    if method == "random":
        sample_size = runtime.get("random_count", 1)
        sample_size = max(1, min(sample_size, len(paths)))
        return random.sample(paths, sample_size)
    if method == "iterate":
        chunk = runtime["chunk_size"]
        start = runtime["index"]
        selected = []
        for i in range(chunk):
            idx = (start + i) % len(paths)
            selected.append(paths[idx])
        runtime["index"] = (start + chunk) % len(paths)
        return selected
    return [None]


def get_rotation_ranges(rotation_config, class_name):
    if not rotation_config:
        return None
    cls = class_name.lower()
    if cls in rotation_config and rotation_config[cls]:
        return rotation_config[cls]
    return rotation_config.get("default")


def load_object_properties_config(config_path):
    candidate = None
    if config_path:
        candidate = Path(config_path)
    else:
        candidate = _default_config_path("object_properties.json")
    if candidate is None:
        return None
    if not candidate.is_file():
        print(f"Object property config file '{candidate}' not found, ignoring.")
        return None
    try:
        data = json.loads(candidate.read_text())
    except Exception as exc:
        print(f"Failed to read object property config '{candidate}': {exc}")
        return None

    normalized = {}
    def normalize_emissive(entry):
        if isinstance(entry, dict):
            return [entry]
        elif isinstance(entry, list):
            return entry
        return None

    def normalize_glare_settings(glare_value):
        defaults = {
            "enabled": False,
            "threshold": 0.7,
            "mix": 0.0,
            "size": 6
        }
        if isinstance(glare_value, dict):
            normalized = defaults.copy()
            for key in defaults:
                if glare_value.get(key) is not None:
                    normalized[key] = glare_value[key]
            normalized["enabled"] = bool(normalized["enabled"])
            return normalized
        if isinstance(glare_value, bool):
            normalized = defaults.copy()
            normalized["enabled"] = glare_value
            return normalized
        return None

    for cls, props in data.items():
        if not isinstance(props, dict):
            continue
        entry = {}
        emissive = props.get("emissive") or props.get("Emissive")
        emissive = normalize_emissive(emissive)
        if emissive:
            cleaned = []
            for item in emissive:
                if not isinstance(item, dict):
                    continue
                cleaned.append({
                    "body": item.get("body"),
                    "color": item.get("color"),
                    "color_list": item.get("color_list"),
                    "intensity": item.get("intensity", 5.0),
                    "alpha": item.get("alpha", 1.0),
                    "status": item.get("status", "on"),
                    "glare": normalize_glare_settings(item.get("glare"))
                })
            if cleaned:
                entry["emissive"] = cleaned
        if entry:
            normalized[cls.lower()] = entry
    return normalized if normalized else None

def setup_compositor_glare(glare_settings=None):
    """
    Set up Blender's compositor to add a glare/bloom effect based on bright pixels.

    glare_settings: dict with keys:
        enabled   -> if False or missing, compositor stays off.
        threshold -> brightness threshold for glare (lower = more glow).
        mix       -> Blender's glare mix (-1 original, 0 mix, 1 glare only).
        size      -> glare radius (Blender accepts 1..9 but we clamp 1..9).
    """
    settings = glare_settings or {}
    enable_glare = bool(settings.get("enabled"))
    threshold = settings.get("threshold", 0.7)
    mix = settings.get("mix", 0.0)
    size = settings.get("size", 6)
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        threshold = 0.7
    try:
        mix = float(mix)
    except (TypeError, ValueError):
        mix = 0.0
    try:
        size = int(round(float(size)))
    except (TypeError, ValueError):
        size = 6
    size = max(1, min(9, size))

    scene = bpy.context.scene

    if not enable_glare:
        # Use default (no compositor) if glare is disabled
        scene.use_nodes = False
        return

    scene.use_nodes = True
    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links

    # Clear any existing compositor setup
    nodes.clear()

    # Render Layers node (input from renderer)
    rl = nodes.new(type="CompositorNodeRLayers")
    rl.location = (-300, 0)

    # Glare node
    glare = nodes.new(type="CompositorNodeGlare")
    glare.location = (0, 0)
    glare.glare_type = 'FOG_GLOW'   # nice soft bloom
    glare.quality = 'HIGH'
    glare.threshold = threshold
    glare.mix = mix
    glare.size = size

    # Composite output (final image)
    comp = nodes.new(type="CompositorNodeComposite")
    comp.location = (300, 0)

    # Wire them up: RenderLayers -> Glare -> Composite
    links.new(rl.outputs["Image"], glare.inputs["Image"])
    links.new(glare.outputs["Image"], comp.inputs["Image"])

def get_object_properties(properties_config, class_name):
    if not properties_config:
        return None
    cls = class_name.lower()
    base = None
    if cls in properties_config:
        base = properties_config[cls]
    elif "default" in properties_config:
        base = properties_config["default"]
    if base:
        return copy.deepcopy(base)
    return None


_missing_emissive_targets = set()

def pick_emissive_color(entry):
    color_spec = entry.get("color")
    color_list = entry.get("color_list")
    def ensure_rgba(values):
        vals = list(values)
        if len(vals) >= 4:
            return vals[:4]
        while len(vals) < 3:
            vals.append(0.0)
        vals = vals[:3]
        vals.append(entry.get("alpha", 1.0))
        return vals

    if isinstance(color_spec, (list, tuple)) and len(color_spec) >= 3:
        return ensure_rgba(color_spec)
    if isinstance(color_spec, str):
        if color_spec.lower() == "random":
            color_spec = None
        else:
            # Unrecognized string, fall back to random below
            color_spec = None

    if isinstance(color_list, list):
        valid_choices = [c for c in color_list if isinstance(c, (list, tuple)) and len(c) >= 3]
        if valid_choices:
            choice = random.choice(valid_choices)
            return ensure_rgba(choice)
    return ensure_rgba([random.random(), random.random(), random.random()])

def should_enable_emissive(entry):
    status = entry.get("status", "on")
    if isinstance(status, str):
        normalized = status.strip().lower()
    else:
        normalized = "on"
    if normalized == "on":
        return True
    if normalized == "off":
        return False
    if normalized == "random":
        return bool(random.getrandbits(1))
    return True

def pick_emissive_intensity(entry):
    value = entry.get("intensity", 5.0)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            low = float(value[0])
            high = float(value[1])
        except (TypeError, ValueError):
            low = high = 5.0
        if low > high:
            low, high = high, low
        return random.uniform(low, high)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 5.0


def apply_object_properties(subparts, properties, class_name):
    if not properties:
        return
    emissive_cfg = properties.get("emissive")
    if emissive_cfg:
        for entry in emissive_cfg:
            apply_emissive_property(subparts, entry, class_name)

def apply_emissive_property(subparts, entry, class_name):
    if not should_enable_emissive(entry):
        return
    body = entry.get("body")
    targets = []
    if body:
        body_lower = body.lower()
        for obj in subparts:
            name = obj.get_name() or ""
            if body_lower in name.lower():
                targets.append(obj)
        if not targets:
            key = (class_name.lower(), body_lower)
            if key not in _missing_emissive_targets:
                print(f"Warning: emissive body '{body}' not found in model '{class_name}'.")
                _missing_emissive_targets.add(key)
            return
    else:
        targets = subparts

    color = pick_emissive_color(entry)           # RGBA list
    strength = pick_emissive_intensity(entry)

    for obj in targets:
        # Try to reuse existing material if there is one
        mats = obj.get_materials()
        if mats:
            mat = mats[0]
        else:
            # Create a new material and assign it
            mat_name = f"emissive_{class_name}_{body or 'all'}"
            mat = bp.material.create(mat_name)
            obj.set_material(0, mat)

        # Make it emissive with the chosen color
        mat.make_emissive(emission_strength=strength, emission_color=color)
        # Optional: if you want the base color of the non-emissive part to match as well:
        # mat.set_principled_shader_value("Base Color", color)


def rotated_rectangle_extents(w, h, angle):
    """
    Given a rectangle of size W x H that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if w <= 0 or h <= 0:
        return 0,0

    width_is_longer = w >= h
    side_long, side_short = (w,h) if width_is_longer else (h,w)

    # since the solutions for angle, -angle and 180-angle are all the same, it
    # suffices to only look at the first quadrant and the absolute values of
    # sin,cos:
    sin_a, cos_a = abs(sin(angle)), abs(cos(angle))
    if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short
        wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

    return wr,hr


def crop_around_center(image, width, height):
    """
    Crop 'image' (a PIL image) to 'width' and height' around the images center
    point
    """
    size = image.size
    center = (int(size[0] * 0.5), int(size[1] * 0.5))

    if(width > size[0]):
        width = size[0]

    if(height > size[1]):
        height = size[1]

    x1 = int(center[0] - width * 0.5)
    x2 = int(center[0] + width * 0.5)
    y1 = int(center[1] - height * 0.5)
    y2 = int(center[1] + height * 0.5)

    return image.crop((x1, y1, x2, y2)) # (left, upper, right, lower)


def crop_to_rotation(img, angle):
    # 'img' is a PIL Image of uint8 RGB values
    # 'angle' is in degrees
    angle_rad = angle*pi/180.0
    width, height = img.size

    img = img.rotate(angle)
    # Crop out black border resulting from rotation
    wr, hr = rotated_rectangle_extents(width, height, angle_rad)
    return crop_around_center(img, wr, hr)


def scale_to_original_shape(img, o_width, o_height):
    c_width, c_height = img.size
    o_ar = o_width/o_height
    c_ar = c_width/c_height
    if o_ar > c_ar:
        cropped = crop_around_center(img, c_width, c_width/o_ar)
    else:
        cropped = crop_around_center(img, c_height*o_ar, c_height)

    return cropped.resize((o_width, o_height))


def get_cuboid_image_space(mesh, camera):
    # object aligned bounding box coordinates in world coordinates
    bbox = mesh.get_bound_box()
    '''
    bbox is a list of the world-space coordinates of the corners of a
    blender object's oriented bounding box
     https://blender.stackexchange.com/questions/32283/what-are-all-values-in-bound-box
    The points from Blender are ordered like so:


            2 +-----------------+ 6
             /|                /|
            /                 / |
         1 +-----------------+ 5
           |     z    y      |  |
           |      | /        |  |
           |      |/         |  |
           |  |   *--- x     |  |
            3 +------------  |  + 7
           | /               | /
           |                 |/
         0 +-----------------+ 4

      But these points must be arranged to match the NVISII ordering
      (Deep_Object_Pose/data_generation/nvisii_data_gen/utils.py:927)

           3 +-----------------+ 0
            /                 /|
           /                 / |
        2 +-----------------+ 1|
          |     z    x      |  |
          |       | /       |  |
          |       |/        |  |
          |  y <--*         |  |
          | 7 +----         |  + 4
          |  /              | /
          | /               |/
        6 +-----------------+ 5

    '''

    centroid = np.array([0.,0.,0.])
    for ii in range(8):
        centroid += bbox[ii]
    centroid = centroid / 8

    cam_pose = np.linalg.inv(camera.get_camera_pose()) # 4x4 world to camera transformation matrx.
    # rvec & tvec describe the world to camera coordinate system
    tvec = -cam_pose[0:3,3]
    rvec = -cv2.Rodrigues(cam_pose[0:3,0:3])[0]
    K = camera.get_intrinsics_as_K_matrix()

    # However these points are in a different order than the original DOPE data format,
    # so we must reorder them (including coordinate frame changes)
    dope_order = [5, 1, 2, 6, 4, 0, 3, 7]

    cuboid = [None for ii in range(9)]
    for ii in range(8):
        cuboid[dope_order[ii]] = cv2.projectPoints(bbox[ii], rvec, tvec, K, np.array([]))[0][0][0]
    cuboid[8] = cv2.projectPoints(centroid, rvec, tvec, K, np.array([]))[0][0][0]

    return np.array(cuboid, dtype=float).tolist()


def write_json(outf, args, camera, objects, objects_data, seg_map):
    cam_xform = camera.get_camera_pose()
    eye = -cam_xform[0:3,3]
    at = -cam_xform[0:3,2]
    up = cam_xform[0:3,0]

    K = camera.get_intrinsics_as_K_matrix()

    data = {
        "camera_data" : {
            "width" : args.width,
            'height' : args.height,
            'camera_look_at':
            {
                'at': [
                    at[0],
                    at[1],
                    at[2],
                ],
                'eye': [
                    eye[0],
                    eye[1],
                    eye[2],
            ],
                'up': [
                    up[0],
                    up[1],
                    up[2],
                ]
            },
            'intrinsics':{
                'fx':K[0][0],
                'fy':K[1][1],
                'cx':K[0][2],
                'cy':K[1][2]
            }
        },
        "objects" : []
    }

    ## Object data
    ##
    for ii, oo in enumerate(objects):
        idx = ii+1 # objects ID indices start at '1'

        num_pixels = int(np.sum((seg_map == idx)))

        if num_pixels < args.min_pixels:
            continue
        projected_keypoints = get_cuboid_image_space(oo, camera)

        data['objects'].append({
            'class': objects_data[ii]['class'],
            'name': objects_data[ii]['name'],
            'visibility': num_pixels,
            'projected_cuboid': projected_keypoints,
            ## 'location' and 'quaternion_xyzw' are both optional data fields,
            ## not used for training
            'location': objects_data[ii]['location'],
            'quaternion_xyzw': objects_data[ii]['quaternion_xyzw']
        })

    with open(outf, "w") as write_file:
        json.dump(data, write_file, indent=4)

    return data


def draw_cuboid_markers(objects, camera, im):
    colors = ['yellow', 'magenta', 'blue', 'red', 'green', 'orange', 'brown', 'cyan', 'white']
    R = 2 # radius
    # draw dots on image to label the cuiboid vertices
    draw = ImageDraw.Draw(im)
    for oo in objects:
        projected_keypoints = get_cuboid_image_space(oo, camera)
        for idx, pp in enumerate(projected_keypoints):
            x = int(pp[0])
            y = int(pp[1])
            draw.ellipse((x-R, y-R, x+R, y+R), fill=colors[idx])

    return im


def randomize_background(path, width, height):
    """Load a background image and scale it to cover the target size without additional transformations."""
    img = Image.open(path).convert("RGB")
    src_w, src_h = img.size
    if src_w == 0 or src_h == 0:
        return img.resize((width, height), Image.BICUBIC)

    scale = max(width / src_w, height / src_h)
    new_size = (int(src_w * scale + 0.5), int(src_h * scale + 0.5))
    img = img.resize(new_size, Image.BICUBIC)

    left = max(0, (img.size[0] - width) // 2)
    top = max(0, (img.size[1] - height) // 2)
    img = img.crop((left, top, left + width, top + height))
    return img


def set_world_background_hdr(filename, strength=1.0, rotation_euler=None):
    """
    Sets the background with a Poly Haven HDRI file

    strength: The brightness of the background.
    rot_euler: Optional euler angles to rotate the background.
    """
    if rotation_euler is None:
        rotation_euler = [0.0, 0.0, 0.0]

    nodes = bpy.context.scene.world.node_tree.nodes
    links = bpy.context.scene.world.node_tree.links

    # add a texture node and load the image and link it
    texture_node = nodes.new(type="ShaderNodeTexEnvironment")
    texture_node.image = bpy.data.images.load(filename, check_existing=True)

    # get the background node of the world shader and link the new texture node
    background_node = Utility.get_the_one_node_with_type(nodes, "Background")
    links.new(texture_node.outputs["Color"], background_node.inputs["Color"])

    # Set the brightness
    background_node.inputs["Strength"].default_value = strength

    # add a mapping node and a texture coordinate node
    mapping_node = nodes.new("ShaderNodeMapping")
    tex_coords_node = nodes.new("ShaderNodeTexCoord")

    #link the texture coordinate node to mapping node and vice verse
    links.new(tex_coords_node.outputs["Generated"], mapping_node.inputs["Vector"])
    links.new(mapping_node.outputs["Vector"], texture_node.inputs["Vector"])

    mapping_node.inputs["Rotation"].default_value = rotation_euler


def main(args):
    ## Segmentation values
    SEG_DISTRACT = 0

    ## All units used are in centimeters

    # Make output directories
    out_directory = os.path.join(args.outf, str(args.run_id))
    os.makedirs(out_directory, exist_ok=True)

    sence_config_data = load_sence_config_data(args.rotation_config)
    rotation_config = load_rotation_config(args.rotation_config, config_data=sence_config_data)
    hdri_background_config = load_hdri_background_config(args.rotation_config, config_data=sence_config_data)
    background_overlay_config = load_background_overlay_config(args.rotation_config, config_data=sence_config_data)
    object_properties_config = load_object_properties_config(args.object_properties)

    # Construct list of background images
    image_types = ('*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG', '*.hdr', '*.HDR')
    backdrop_images = []
    background_idx = 0  # index for cycling through backgrounds

    hdr_backdrops = []
    if args.backgrounds_folder is not None:
        for ext in image_types:
            backdrop_images.extend(glob.glob(os.path.join(args.backgrounds_folder,
                                                          os.path.join('**', ext)),
                                             recursive=True))

        # Sort for deterministic order (optional but nice for reproducibility)
        backdrop_images.sort()

        if len(backdrop_images) == 0:
            print(f"No images found in backgrounds directory '{args.backgrounds_folder}'")
        else:
            num_hdr = sum(
                1 for p in backdrop_images
                if os.path.splitext(p)[1].lower() in (".hdr", ".exr")
            )
            num_ldr = len(backdrop_images) - num_hdr
            print(
                f"{len(backdrop_images)} background images found in '{args.backgrounds_folder}' "
                f"({num_hdr} HDR / {num_ldr} non-HDR). "
                "Backgrounds cycle sequentially by default; HDR-specific settings in sence_config can override this."
            )
        hdr_backdrops = [
            p for p in backdrop_images
            if os.path.splitext(p)[1].lower() in (".hdr", ".exr")
        ]

    hdri_runtime = None
    if hdri_background_config:
        if not hdr_backdrops:
            print("HDRI backgrounds enabled in config but no HDR files were found; ignoring that setting.")
        else:
            hdri_runtime = {
                "paths": hdr_backdrops,
                "method": hdri_background_config["selection_method"],
                "strength_range": hdri_background_config["strength_range"],
                "random_selection": None
            }
            if hdri_runtime["method"] == "random":
                total_hdr = len(hdr_backdrops)
                desired_count = hdri_background_config.get("count")
                if desired_count is None or desired_count == "all":
                    sample_size = total_hdr
                else:
                    try:
                        sample_size = int(desired_count)
                    except (TypeError, ValueError):
                        sample_size = 1
                sample_size = max(1, min(sample_size, total_hdr))
                hdri_runtime["random_selection"] = random.sample(hdr_backdrops, sample_size)

    fallback_overlay_paths = [
        p for p in backdrop_images
        if os.path.splitext(p)[1].lower() not in HDR_FILE_EXTENSIONS
    ]
    fallback_overlay_idx = 0

    background_overlay_runtime = None
    if background_overlay_config:
        dataset_dir = background_overlay_config["folder"]
        overlay_paths = []
        if dataset_dir.is_dir():
            for candidate in dataset_dir.rglob("*"):
                if candidate.is_file() and candidate.suffix.lower() in LDR_IMAGE_EXTENSIONS:
                    overlay_paths.append(str(candidate))
            overlay_paths.sort()
            if overlay_paths:
                requested_count = background_overlay_config["count"]
                total_available = len(overlay_paths)
                if requested_count is None or requested_count < 1 or requested_count > total_available:
                    requested_count = total_available
                method = background_overlay_config["selection_method"]
                if method == "iterate":
                    background_overlay_runtime = {
                        "paths": overlay_paths,
                        "method": method,
                        "chunk_size": requested_count,
                        "index": 0
                    }
                else:
                    background_overlay_runtime = {
                        "paths": overlay_paths,
                        "method": "random",
                        "random_count": requested_count
                    }
            else:
                print(f"Backgrounds config enabled but no images found under '{dataset_dir}'.")
        else:
            print(f"Backgrounds config enabled but folder '{dataset_dir}' does not exist.")

    frames_per_cycle = max(1, int(args.frames_per_cycle))
    if hdri_runtime:
        if hdri_runtime["method"] == "iterate":
            hdr_multiplier = len(hdri_runtime["paths"])
        else:
            hdr_multiplier = len(hdri_runtime["random_selection"]) if hdri_runtime["random_selection"] else 1
    else:
        hdr_multiplier = 1
    if background_overlay_runtime:
        if background_overlay_runtime["method"] == "iterate":
            background_multiplier = background_overlay_runtime["chunk_size"]
        else:
            background_multiplier = background_overlay_runtime["random_count"]
    else:
        background_multiplier = 1
    total_frames = frames_per_cycle * hdr_multiplier * background_multiplier

    def next_mixed_backdrop():
        nonlocal background_idx
        if not backdrop_images:
            return (None, None)
        path = backdrop_images[background_idx]
        background_idx = (background_idx + 1) % len(backdrop_images)
        ext = os.path.splitext(path)[1].lower()
        if ext in HDR_FILE_EXTENSIONS:
            return (path, None)
        return (None, path)

    def next_fallback_overlay():
        nonlocal fallback_overlay_idx
        if not fallback_overlay_paths:
            return None
        path = fallback_overlay_paths[fallback_overlay_idx]
        fallback_overlay_idx = (fallback_overlay_idx + 1) % len(fallback_overlay_paths)
        return path

    # Construct list of object models
    object_models = []
    tmp_p = None
    if args.path_single_obj:
        object_models.append(args.path_single_obj)
        tmp_p = args.path_single_obj
    else:
        object_models = glob.glob(args.objs_folder + "**/textured.obj", recursive=True)
        tmp_p = args.objs_folder
    if len(object_models) == 0:
        print(f"Failed to find any loadable models at {tmp_p}")
        exit(1)

    # Construct list of distractors
    distractor_objs = glob.glob(args.distractors_folder + "**/model.obj", recursive=True)
    print(f"{len(distractor_objs)} distractor objects found.")

    # Set up blenderproc
    bp.init()
    
    # Make world background invisible to the camera, but keep it for lighting/reflections
    bpy.context.scene.render.film_transparent = True

    # Set the camera to be in front of the object
    cam_pose = bp.math.build_transformation_mat([0, -25, 0], [np.pi / 2, 0, 0])
    bp.camera.add_camera_pose(cam_pose)
    bp.camera.set_resolution(args.width, args.height)
    if args.focal_length:
        K = np.array([[args.focal_length, 0, args.width/2],
                      [0, args.focal_length, args.height/2],
                      [0,0,1]])
        bp.camera.set_intrinsics_from_K_matrix(K, args.width, args.height, clip_start=1.0,
                                               clip_end=1000.0)
    else:
        bp.camera.set_intrinsics_from_blender_params(lens=0.785398, # FOV in radians
                                                     lens_unit='FOV',
                                                     clip_start=1.0, clip_end=1000.0)

    # Create lights
    #bp.renderer.set_world_background([1,1,1], 1.0)
    #static_light = bp.types.Light()
    #static_light.set_type('SUN')
    #static_light.set_energy(1) # watts per sq. meter

    #light = bp.types.Light()
    #light.set_type('POINT')
    #light.set_energy(100) # watts per sq. meter

    light = bp.lighting.add_intersecting_spot_lights_to_camera_poses(5.0, 50.0)


    # Renderer setup
    bp.renderer.set_output_format('PNG', enable_transparency=True)
    bp.renderer.set_render_devices(desired_gpu_ids=[0])


    # Create objects
    objects = []
    object_subparts = []
    object_property_settings = []
    objects_data = []
    for idx in range(args.nb_objects):
        model_path =  object_models[random.randint(0, len(object_models) - 1)]
        loaded_objs = bp.loader.load_obj(
            model_path,
            use_split_groups=True,
            use_split_objects=True
        )
        obj_class = args.object_class
        if len(loaded_objs) == 0:
            print(f"Warning: no meshes loaded from {model_path}")
            continue
        print(f"Loaded object '{obj_class}' with parts:")
        for part in loaded_objs:
            try:
                print(f"  - {part.get_name()}")
            except Exception:
                print("  - (unnamed part)")
        obj = loaded_objs[0]
        obj.set_cp("category_id", 1+idx)
        objects.append(obj)
        object_subparts.append(loaded_objs)
        obj_name = obj_class + "_" + str(idx).zfill(3)
        objects_data.append({'class': obj_class,
                            'name': obj_name,
                            'id':1+idx
                            })
        object_property_settings.append(get_object_properties(object_properties_config, obj_class))
        
    # Decide if we need glare based on object properties (but don't enable yet)
    glare_settings = None
    for settings in object_property_settings:
        if not settings:
            continue
        emissive = settings.get("emissive", [])
        for entry in emissive:
            glare_entry = entry.get("glare")
            if isinstance(glare_entry, dict) and glare_entry.get("enabled"):
                glare_settings = glare_entry
                break
        if glare_settings:
            break

    # Create distractor(s)
    distractors = []
    if len(distractor_objs) > 0:
        for idx_obj in range(int(args.nb_distractors)):
            distractor_fn = distractor_objs[random.randint(0,len(distractor_objs)-1)]
            distractor = bp.loader.load_obj(distractor_fn)[0]
            distractor.set_cp("category_id", SEG_DISTRACT)
            distractors.append(distractor)
            print(f"loaded {distractor_fn}")

    frame_counter = 0
    if hdri_runtime:
        if hdri_runtime["method"] == "random":
            hdr_sequence = hdri_runtime["random_selection"] or []
        else:
            hdr_sequence = hdri_runtime["paths"]
    else:
        hdr_sequence = [None]

    for hdr_path in hdr_sequence:
        overlay_candidates = select_backgrounds_for_hdr(background_overlay_runtime) if background_overlay_runtime else None
        for _ in range(frames_per_cycle):
            env_path = hdr_path
            fallback_overlay = None
            if hdri_runtime is None:
                fallback_env, fallback_overlay = next_mixed_backdrop()
                if env_path is None:
                    env_path = fallback_env
            else:
                fallback_overlay = next_fallback_overlay()

            if background_overlay_runtime:
                overlays_to_apply = overlay_candidates or [fallback_overlay]
            else:
                overlays_to_apply = [fallback_overlay]

            overlays_to_apply = overlays_to_apply or [None]

            # Place object(s)
            for idx, oo in enumerate(objects):
                xform = np.eye(4)
                xform[0:3, 3] = random_object_position(near=20, far=100)
                rot_ranges = get_rotation_ranges(rotation_config, objects_data[idx]['class'])
                xform[0:3, 0:3] = random_rotation_matrix(rot_ranges)

                for part in object_subparts[idx]:
                    part.set_local2world_mat(xform)
                    part.set_scale([args.scale, args.scale, args.scale])

                xform_in_cam = np.linalg.inv(bp.camera.get_camera_pose()) @ xform
                objects_data[idx]['location'] = xform_in_cam[0:3, 3].tolist()
                tmp_wxyz = Quaternion(matrix=xform_in_cam[0:3, 0:3]).elements
                q_xyzw = [tmp_wxyz[1], tmp_wxyz[2], tmp_wxyz[3], tmp_wxyz[0]]
                objects_data[idx]['quaternion_xyzw'] = q_xyzw

            for idx in range(len(objects)):
                apply_object_properties(object_subparts[idx], object_property_settings[idx], objects_data[idx]['class'])

            for dd in distractors:
                xform = np.eye(4)
                xform[0:3,3] = point_in_frustrum(bp.camera, near=5.0, far=100.)
                xform[0:3,0:3] = random_rotation_matrix()
                dd.set_local2world_mat(xform)
                dd.set_scale([args.distractor_scale, args.distractor_scale, args.distractor_scale])

            # Configure HDR environment if available
            env_is_hdr = env_path and os.path.splitext(env_path)[1].lower() in HDR_FILE_EXTENSIONS
            if env_is_hdr:
                if hdri_runtime and env_path in hdri_runtime["paths"]:
                    strength_range = hdri_runtime["strength_range"]
                    method_label = hdri_runtime["method"]
                else:
                    strength_range = (0.5, 1.5)
                    method_label = "sequential"
                strength = random.uniform(strength_range[0], strength_range[1])
                rotation = [
                    random.random() * 0.2 - 0.1,
                    random.random() * 0.2 - 0.1,
                    random.random() * 0.2 - 0.1,
                ]
                set_world_background_hdr(env_path, strength, rotation)
                print(
                    f"Run {args.run_id}: HDR '{os.path.basename(env_path)}' "
                    f"(strength={strength:.2f}, method={method_label})",
                    flush=True,
                )

            # redirect blenderproc output to log file
            logfile = '/tmp/blender_render.log'
            open(logfile, 'a').close()
            old = os.dup(sys.stdout.fileno())
            sys.stdout.flush()
            os.close(sys.stdout.fileno())
            fd = os.open(logfile, os.O_WRONLY)

            setup_compositor_glare({"enabled": False})
            segs = bp.renderer.render_segmap()
            captured_segmaps = segs['class_segmaps'][0]

            setup_compositor_glare(glare_settings)
            data = bp.renderer.render()

            os.close(fd)
            os.dup(old)
            os.close(old)

            base_image = Image.fromarray(data['colors'][0])

            for overlay_path in overlays_to_apply:
                frame_id = frame_counter
                frame_counter += 1
                print(f"Run {args.run_id}: {frame_counter}/{total_frames}", flush=True)

                im = base_image.copy()

                if overlay_path:
                    background = randomize_background(overlay_path, args.width, args.height)
                    background = background.convert('RGB')
                    background.paste(im, mask=im.convert('RGBA'))
                    im = background

                if args.debug:
                    im = draw_cuboid_markers(objects, bp.camera, im)

                file_base = os.path.join(out_directory, f"{frame_id:06d}")
                im.save(file_base + ".png")
                write_json(file_base + ".json", args, bp.camera, objects, objects_data, captured_segmaps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Parameters passed from run script; they are ignored here
    parser.add_argument(
        '--nb_runs',
        default=1,
        type=int,
        help='Number of times the datagen script is run. Each time it is run, a new set of '
        'distractors is selected.'
    )
    ## Parameters for this script
    parser.add_argument(
        '--run_id',
        default=0,
        type=int,
        help='Output files will be put in a subdirectory of this name. This parameter should '
        'not be set by the user'
    )
    parser.add_argument(
        '--width',
        default=500,
        type=int,
        help = 'image output width'
    )
    parser.add_argument(
        '--height',
        default=500,
        type=int,
        help = 'image output height'
    )
    parser.add_argument(
        '--focal-length',
        default=None,
        type=float,
        help = "focal length of the camera"
    )
    parser.add_argument(
        '--distractors_folder',
        default='google_scanned_models/',
        help = "folder containing distraction objects"
    )
    parser.add_argument(
        '--objs_folder',
        default='models/',
        help = "folder containing training objects, if using multiple"
    )
    parser.add_argument(
        '--path_single_obj',
        default=None,
        help='If you have a single obj file, path to the obj directly.'
    )
    parser.add_argument(
        '--object_class',
        required=True,
        help="The class name of the object(s) you will be training to recognize."
    )
    parser.add_argument(
        '--scale',
        default=1,
        type=float,
        help='Scaling to apply to the target object(s) to put in units of centimeters; e.g if '
             'the object scale is meters -> scale=0.01; if it is in cm -> scale=1.0'
    )
    parser.add_argument(
        '--backgrounds_folder',
        default=None,
        help = "folder containing background images. Images can .jpeg, .png, or .hdr."
    )
    parser.add_argument(
        '--nb_objects',
        default=1,
        type = int,
        help = "how many objects"
    )
    parser.add_argument(
        '--nb_distractors',
        default=1,
        help = "how many distractor objects"
    )
    parser.add_argument(
        '--distractor_scale',
        default=50,
        type=float,
        help='Scaling to apply to distractor objects in order to put in units of centimeters; '
             'e.g if the object scale is meters -> scale=100; if it is in cm -> scale=1'
    )
    parser.add_argument(
        '--frames_per_cycle', '--nb_frames',
        dest='frames_per_cycle',
        type=int,
        default=2000,
        help="How many frames to render per background/HDR cycle (alias --nb_frames). "
             "When HDRI selection_method='iterate', total frames = frames_per_cycle * number_of_HDRs."
    )
    parser.add_argument(
        '--min_pixels',
        type = int,
        default=1,
        help = "How many visible pixels an object must have to be included in the JSON data"
    )
    parser.add_argument(
        '--outf',
        default='output_example/',
        help = "output filename inside output/"
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help="Render the cuboid corners as small spheres. Only for debugging purposes;"
        "do not use for training!"
    )
    parser.add_argument(
        '--rotation_config',
        default=None,
        help="Path to a JSON file specifying per-class rotation limits (degrees). "
             "If omitted, the script looks for 'sence_config.json' under the repo 'config' directory."
    )
    parser.add_argument(
        '--object_properties',
        default=None,
        help="Path to JSON describing per-object properties (e.g., emissive parts). "
             "Defaults to 'object_properties.json' inside the repo-wide 'config' directory."
    )

    opt = parser.parse_args()
    main(opt)
