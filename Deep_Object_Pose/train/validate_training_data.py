#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
COMMON_DIR = ROOT_DIR / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))


def _normalize_extension(ext):
    ext = ext.lower()
    if not ext.startswith("."):
        return f".{ext}"
    return ext


def _format_extensions(exts):
    return tuple(_normalize_extension(ext) for ext in exts)


def _estimate_total_frames(datasets, extensions):
    total = 0
    for dataset in datasets:
        abs_path = os.path.abspath(dataset)
        for root, _, files in os.walk(abs_path):
            max_idx = 0
            for file in files:
                if not file.lower().endswith(extensions):
                    continue
                digits = "".join(ch for ch in Path(file).stem if ch.isdigit())
                if digits:
                    try:
                        max_idx = max(max_idx, int(digits))
                    except ValueError:
                        continue
            total += max_idx
    return total


def _print_progress(current, total, prefix="Scanning"):
    total = max(total, 1)
    percent = min(current / total, 1.0)
    bar_length = 40
    filled = int(bar_length * percent)
    bar = "#" * filled + "-" * (bar_length - filled)
    sys.stdout.write(
        f"\r{prefix}: [{bar}] {current}/{total} ({percent * 100:5.1f}%)"
    )
    sys.stdout.flush()


def _verify_json_files(entries):
    """Optionally ensure every JSON file is readable."""
    bad_files = []
    total = len(entries)
    for idx, (_, _, json_path) in enumerate(entries, 1):
        try:
            with open(json_path, "r") as fh:
                json.load(fh)
        except Exception as exc:
            print(f"[ERROR] Failed to parse '{json_path}': {exc}")
            bad_files.append(json_path)
        _print_progress(idx, total, prefix="Verifying JSON")
    print()
    if bad_files:
        raise RuntimeError(
            f"{len(bad_files)} annotation file(s) could not be read. "
            "See messages above for details."
        )


def _collect_entries(dataset_paths, extensions):
    normalized_exts = _format_extensions(extensions)
    estimated_total = _estimate_total_frames(dataset_paths, normalized_exts)
    processed = 0
    all_entries = []
    for dataset in dataset_paths:
        abs_dataset = os.path.abspath(dataset)
        dataset_start = len(all_entries)
        for root, _, files in os.walk(abs_dataset):
            for file in files:
                if not file.lower().endswith(normalized_exts):
                    continue
                img_path = os.path.join(root, file)
                json_path = os.path.splitext(img_path)[0] + ".json"
                processed += 1
                _print_progress(processed, estimated_total)
                if os.path.isfile(json_path):
                    all_entries.append((img_path, file, json_path))
        dataset_count = len(all_entries) - dataset_start
        print(f"\n{abs_dataset}: {dataset_count} image(s) with annotations found.")
    print()
    return all_entries


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate local DOPE training data and optionally write a manifest "
            "file that train.py can consume via --data_manifest."
        )
    )
    parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="Path(s) to the training dataset directories.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="training_manifest.json",
        help="Where to write the manifest JSON (default: training_manifest.json).",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=["png"],
        help="Image extensions to scan for (default: png).",
    )
    parser.add_argument(
        "--verify-json",
        action="store_true",
        help="Open every annotation JSON to ensure it is parseable.",
    )
    args = parser.parse_args()

    entries = _collect_entries(args.data, extensions=args.extensions)
    if not entries:
        print("No training samples were discovered. Nothing to validate.")
        return

    if args.verify_json:
        _verify_json_files(entries)

    manifest = [
        {
            "image_path": img_path,
            "json_path": json_path,
            "image_name": img_name,
        }
        for (img_path, img_name, json_path) in entries
    ]

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2))
    print(
        f"Saved manifest with {len(manifest)} entries to '{output_path}'. "
        "Pass this path to train.py via --data_manifest to skip the directory scan."
    )


if __name__ == "__main__":
    main()
