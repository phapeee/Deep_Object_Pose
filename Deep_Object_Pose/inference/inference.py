#!/usr/bin/env python

import argparse
import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import simplejson as json
import sys
import time
import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMON_DIR = os.path.join(ROOT_DIR, "common")
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

from cuboid import Cuboid3d
from cuboid_pnp_solver import CuboidPNPSolver
from detector import ModelData, ObjectDetector
from utils import loadimages_inference, loadweights, Draw
from pyrr import Quaternion


FRAME_CONVERSION_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=float,
)
FRAME_CONVERSION_ROTATION = Quaternion.from_x_rotation(np.pi)


class DopeNode(object):
    """ROS node that listens to image topic, runs DOPE, and publishes DOPE results"""

    def __init__(
        self,
        config,   # config yaml loaded eg dict
        weight,   # path to weight file
        parallel, # was it trained using DDP
        class_name,
        silence_detection=False,
    ):
        self.input_is_rectified = config["input_is_rectified"]
        self.downscale_height = config["downscale_height"]

        self.config_detect = lambda: None
        self.config_detect.mask_edges = 1
        self.config_detect.mask_faces = 1
        self.config_detect.vertex = 1
        self.config_detect.threshold = 0.5
        self.config_detect.softmax = 1000
        self.config_detect.thresh_angle = config["thresh_angle"]
        self.config_detect.thresh_map = config["thresh_map"]
        self.config_detect.sigma = config["sigma"]
        self.config_detect.thresh_points = config["thresh_points"]

        # load network model, create PNP solver
        self.model = ModelData(
            name=class_name,
            net_path=weight,
            parallel=parallel
        )
        self.model.load_net_model()
        print("Model Loaded")

        try:
            self.draw_color = tuple(config["draw_colors"][class_name])
        except:
            self.draw_color = (0, 255, 0)

        self.dimension = tuple(config["dimensions"][class_name])
        self.class_id = config["class_ids"][class_name]

        self.pnp_solver = CuboidPNPSolver(
            class_name, cuboid3d=Cuboid3d(config["dimensions"][class_name])
        )
        self.class_name = class_name
        self.silence_detection = silence_detection

        print("Ctrl-C to stop")

    def image_callback(
        self,
        img,
        camera_info,
        img_name,  # this is the name of the img file to save, it needs the .png at the end
        output_folder,  # folder where to put the output
        weight,
        debug=False,
        save_outputs=True,
        return_image=False
    ):
        # Update camera matrix and distortion coefficients
        if self.input_is_rectified:
            P = np.matrix(
                camera_info["projection_matrix"]["data"], dtype="float64"
            ).copy()
            P.resize((3, 4))
            camera_matrix = P[:, :3]
            dist_coeffs = np.zeros((4, 1))
        else:
            # TODO
            camera_matrix = np.matrix(camera_info.K, dtype="float64")
            camera_matrix.resize((3, 3))
            dist_coeffs = np.matrix(camera_info.D, dtype="float64")
            dist_coeffs.resize((len(camera_info.D), 1))

        # Downscale image if necessary
        height, width, _ = img.shape
        scaling_factor = float(self.downscale_height) / height
        if scaling_factor < 1.0:
            camera_matrix[:2] *= scaling_factor
            img = cv2.resize(
                img, (int(scaling_factor * width), int(scaling_factor * height))
            )

        self.pnp_solver.set_camera_intrinsic_matrix(camera_matrix)
        self.pnp_solver.set_dist_coeffs(dist_coeffs)

        # Copy and draw image
        img_copy = img.copy()
        im = Image.fromarray(img_copy)
        draw = Draw(im)

        # dictionary for the final output
        dict_out = {"camera_data": {}, "objects": []}

        # Detect object
        results, belief_imgs = ObjectDetector.detect_object_in_image(
            self.model.net, self.pnp_solver, img, self.config_detect,
            grid_belief_debug=debug,
            silent=self.silence_detection
        )

        # Publish pose and overlay cube on image
        for _, result in enumerate(results):
            if result["location"] is None:
                continue

            loc = np.array(result["location"], dtype=float)
            ori = result["quaternion"]
            if not isinstance(ori, Quaternion):
                ori = Quaternion(ori)

            # Convert from the OpenCV camera frame (returned by PnP)
            # into the NDDS/Blender frame used by the training data.
            loc = FRAME_CONVERSION_MATRIX.dot(loc)
            ori = FRAME_CONVERSION_ROTATION * ori

            dict_out["objects"].append(
                {
                    "class": self.class_name,
                    "location": loc.tolist(),
                    "quaternion_xyzw": np.array(ori).tolist(),
                    "projected_cuboid": np.array(result["projected_points"]).tolist(),
                }
            )

            # Draw the cube
            if None not in result["projected_points"]:
                points2d = []
                for pair in result["projected_points"]:
                    points2d.append(tuple(pair))
                draw.draw_cube(points2d, self.draw_color)

        if save_outputs:
            # create directory to save image if it does not exist
            img_name_base = img_name.split("/")[-1]
            output_path = os.path.join(
                output_folder,
                weight.split("/")[-1].replace(".pth", ""),
                *img_name.split("/")[:-1],
            )
            if not os.path.isdir(output_path):
                os.makedirs(output_path, exist_ok=True)

            im.save(os.path.join(output_path, img_name_base))
            if belief_imgs is not None:
                belief_imgs.save(os.path.join(output_path, "belief_maps.png"))

            json_path = os.path.join(
                output_path, ".".join(img_name_base.split(".")[:-1]) + ".json"
            )
            # save the json files
            with open(json_path, "w") as fp:
                json.dump(dict_out, fp, indent=2)

        if return_image:
            return np.array(im)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outf",
        default="output",
        help="Where to store the output images and inference results.",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="folder for data images to load.",
    )
    parser.add_argument(
        "--config",
        default="../config/config_pose.yaml",
        help="Path to inference config file",
    )
    parser.add_argument(
        "--camera",
        default="../config/camera_info.yaml",
        help="Path to camera info file",
    )

    parser.add_argument(
        "--weights",
        "--weight",
        "-w",
        required=True,
        help="Path to weights or folder containing weights. If path is to a folder, then script "
        "will run inference with all of the weights in the folder. This could take a while if "
        "the set of test images is large.",
    )

    parser.add_argument(
        "--parallel",
        action='store_true',
        help="Were the weights trained using DDP; if set to true, the names of later weights "
        " will be altered during load to match the model"
    )

    parser.add_argument(
        "--exts",
        nargs="+",
        type=str,
        default=["png"],
        help="Extensions for images to use. Can have multiple entries seperated by space. "
        "e.g. png jpg",
    )

    parser.add_argument(
        "--object",
        required=True,
        help="Name of class to run detections on.",
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help="Generates debugging information, including raw belief maps and annotation of "
        "the results"
    )

    parser.add_argument(
        "--config_dir",
        default=None,
        help="Directory containing config files; if provided, the script will attempt to "
             "load config_pose.yaml and camera_info.yaml (or the first matching config*.yaml / "
             "camera*.yaml) from this folder when explicit --config/--camera values are not supplied."
    )

    parser.add_argument(
        "--webcam",
        type=int,
        default=None,
        help="Index of the webcam to use for live inference. If set, --data is optional."
    )
    parser.add_argument(
        "--webcam_width",
        type=int,
        default=None,
        help="Optional width for the webcam capture."
    )
    parser.add_argument(
        "--webcam_height",
        type=int,
        default=None,
        help="Optional height for the webcam capture."
    )
    parser.add_argument(
        "--webcam_display",
        action='store_true',
        help="Display the live webcam output with detections overlaid."
    )
    parser.add_argument(
        "--webcam_save",
        action='store_true',
        help="Save webcam frames and JSON results to --outf. Disabled by default."
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to a video file for sequential inference."
    )
    parser.add_argument(
        "--video_display",
        action='store_true',
        help="Display processed frames when running on a video file."
    )
    parser.add_argument(
        "--video_save",
        action='store_true',
        help="Save processed video frames and JSON outputs to --outf."
    )
    parser.add_argument(
        "--video_out",
        type=str,
        default=None,
        help="Optional path for writing an annotated video once processing is complete."
    )
    parser.add_argument(
        "--video_out_fps",
        type=float,
        default=None,
        help="Override FPS for --video_out (defaults to source FPS or measured FPS)."
    )
    parser.add_argument(
        "--silence-detection",
        action='store_true',
        help="Suppress intermediate detection logs (e.g., incomplete cuboid warnings)."
    )
    opt = parser.parse_args()

    config_path = Path(opt.config).resolve() if opt.config else None
    camera_path = Path(opt.camera).resolve() if opt.camera else None

    if opt.config_dir is not None:
        config_dir = Path(opt.config_dir)
        if not config_dir.is_dir():
            raise FileNotFoundError(f"Config directory '{config_dir}' does not exist")

        def pick_file(preferred_names, glob_pattern):
            for name in preferred_names:
                candidate = config_dir / name
                if candidate.is_file():
                    return candidate
            matches = sorted(config_dir.glob(glob_pattern))
            if matches:
                return matches[0]
            return None

        if config_path is None or not config_path.is_file():
            candidate = pick_file(["config_pose.yaml"], "*config*.yaml")
            if candidate is None:
                raise FileNotFoundError(
                    f"Could not locate a config YAML inside '{config_dir}'. "
                    "Expected files matching 'config_pose.yaml' or '*config*.yaml'."
                )
            config_path = candidate

        if camera_path is None or not camera_path.is_file():
            candidate = pick_file(
                ["camera_info.yaml", "blenderproc_camera_info_example.yaml"],
                "*camera*.yaml"
            )
            if candidate is None:
                raise FileNotFoundError(
                    f"Could not locate a camera YAML inside '{config_dir}'. "
                    "Expected files matching 'camera_info.yaml', "
                    "'blenderproc_camera_info_example.yaml', or '*camera*.yaml'."
                )
            camera_path = candidate

    if config_path is None or not config_path.is_file():
        raise FileNotFoundError(f"Config file '{config_path}' not found")
    if camera_path is None or not camera_path.is_file():
        raise FileNotFoundError(f"Camera file '{camera_path}' not found")

    # load the configs
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open(camera_path) as f:
        camera_info = yaml.load(f, Loader=yaml.FullLoader)

    os.makedirs(opt.outf, exist_ok=True)

    if opt.webcam is None and opt.video is None and opt.data is None:
        raise ValueError("Please specify --data, --video, or --webcam as an input source.")

    # Load model weights
    weights = loadweights(opt.weights)

    if len(weights) < 1:
        print(
            "No weights found at specified directory. Please check --weights flag and try again."
        )
        exit()
    else:
        print(f"Found {len(weights)} weights. ")

    def process_stream(
        capture,
        stream_name,
        dope_node,
        weight,
        display=False,
        save=False,
        img_prefix="stream",
        collect_frames=False,
        video_out_path=None,
        video_out_fps=None,
    ):
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open {stream_name.lower()}")
        frame_idx = 0
        start_time = time.time()
        last_report = start_time
        window_name = f"DOPE {stream_name}"
        collected_frames = [] if collect_frames else None
        try:
            while True:
                ret, frame_bgr = capture.read()
                if not ret:
                    if stream_name.lower() != "webcam":
                        print(f"{stream_name}: stream ended.")
                    break
                frame_rgb = frame_bgr[..., ::-1].copy()
                img_name = f"{img_prefix}/frame_{frame_idx:06d}.png"
                want_image = display or collect_frames
                output_img = dope_node.image_callback(
                    img=frame_rgb,
                    camera_info=camera_info,
                    img_name=img_name,
                    output_folder=opt.outf,
                    weight=weight,
                    debug=opt.debug,
                    save_outputs=save,
                    return_image=want_image
                )
                frame_idx += 1
                now = time.time()
                if now - last_report >= 1.0:
                    fps = frame_idx / (now - start_time)
                    elapsed = now - start_time
                    print(f"{stream_name}: {frame_idx} frames over {elapsed:.1f}s -> {fps:.2f} FPS", end="\r")
                    last_report = now
                if display and output_img is not None:
                    display_frame = output_img[..., ::-1]
                    cv2.imshow(window_name, display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                if collect_frames and output_img is not None:
                    collected_frames.append(output_img[..., ::-1].copy())
        finally:
            capture.release()
            if display:
                cv2.destroyWindow(window_name)
            if collect_frames and collected_frames:
                out_dir = os.path.dirname(video_out_path) if video_out_path else ""
                if video_out_path:
                    if out_dir:
                        os.makedirs(out_dir, exist_ok=True)
                    source_fps = capture.get(cv2.CAP_PROP_FPS)
                    if not source_fps or source_fps <= 0:
                        elapsed = max(time.time() - start_time, 1e-6)
                        source_fps = frame_idx / elapsed
                    target_fps = video_out_fps or source_fps or 30.0
                    height, width = collected_frames[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(video_out_path, fourcc, target_fps, (width, height))
                    for f in collected_frames:
                        writer.write(f)
                    writer.release()

    if opt.webcam is not None:
        if len(weights) > 1:
            print("Multiple weights detected; using the first weight for webcam inference.")
        weight = weights[0]
        dope_node = DopeNode(config, weight, opt.parallel, opt.object, silence_detection=opt.silence_detection)
        cap = cv2.VideoCapture(opt.webcam)
        if opt.webcam_width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, opt.webcam_width)
        if opt.webcam_height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, opt.webcam_height)
        process_stream(
            cap,
            "Webcam",
            dope_node,
            weight,
            display=opt.webcam_display,
            save=opt.webcam_save,
            img_prefix="webcam"
        )
        print("------")
    elif opt.video is not None:
        if len(weights) > 1:
            print("Multiple weights detected; using the first weight for video inference.")
        weight = weights[0]
        dope_node = DopeNode(config, weight, opt.parallel, opt.object, silence_detection=opt.silence_detection)
        cap = cv2.VideoCapture(opt.video)
        process_stream(
            cap,
            "Video",
            dope_node,
            weight,
            display=opt.video_display,
            save=opt.video_save,
            img_prefix="video",
            collect_frames=bool(opt.video_out),
            video_out_path=opt.video_out,
            video_out_fps=opt.video_out_fps
        )
        print("------")
    else:
        # Load inference images
        imgs, imgsname = loadimages_inference(opt.data, extensions=opt.exts)

        if len(imgs) == 0 or len(imgsname) == 0:
            print(
                "No input images found at specified path and extensions. Please check --data "
                "and --exts flags and try again."
            )
            exit()

        for w_i, weight in enumerate(weights):
            dope_node = DopeNode(config, weight, opt.parallel, opt.object, silence_detection=opt.silence_detection)

            for i in range(len(imgs)):
                print(
                    f"({w_i + 1} of  {len(weights)}) frame {i + 1} of {len(imgs)}: {imgsname[i]}"
                )
                img_name = imgsname[i]

                frame = cv2.imread(imgs[i])

                frame = frame[..., ::-1].copy()

                # call the inference node
                dope_node.image_callback(
                    img=frame,
                    camera_info=camera_info,
                    img_name=img_name,
                    output_folder=opt.outf,
                    weight=weight,
                    debug=opt.debug
                )

            print("------")
