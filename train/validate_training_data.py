#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_METADATA_NAME = ".dope_processed_folders.json"

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


def _estimate_total_frames(datasets, extensions, skip_map=None):
    total = 0
    skip_map = skip_map or {}
    for dataset in datasets:
        abs_path = os.path.abspath(dataset)
        skip_dirs = skip_map.get(abs_path, set())
        for root, dirs, files in os.walk(abs_path):
            if skip_dirs and root == abs_path:
                dirs[:] = [d for d in dirs if d not in skip_dirs]
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


def _load_processed_dirs(metadata_path, reset=False):
    if metadata_path is None or reset:
        return set()
    if not metadata_path.exists():
        return set()
    try:
        payload = json.loads(metadata_path.read_text())
    except Exception as exc:
        print(f"[WARN] Could not parse metadata '{metadata_path}': {exc}")
        return set()
    entries = payload.get("processed_folders", [])
    return {str(item) for item in entries}


def _write_processed_dirs(metadata_path, directories):
    if metadata_path is None:
        return
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps({"processed_folders": sorted(set(directories))}, indent=2)
    )


def _prune_missing_dirs(dataset_root, directories):
    if not directories:
        return set()
    existing = {
        entry.name
        for entry in Path(dataset_root).iterdir()
        if entry.is_dir()
    }
    return {name for name in directories if name in existing}


def _collect_entries(
    dataset_paths,
    extensions,
    metadata_name=DEFAULT_METADATA_NAME,
    reset_metadata=False,
):
    normalized_exts = _format_extensions(extensions)
    metadata_enabled = bool(metadata_name)
    dataset_contexts = []
    for dataset in dataset_paths:
        abs_dataset = os.path.abspath(dataset)
        if not os.path.isdir(abs_dataset):
            print(f"[WARN] '{abs_dataset}' is not a directory. Skipping.")
            continue
        ctx = {
            "abs_path": abs_dataset,
            "metadata_path": None,
            "stored_dirs": set(),
            "processed_dirs": set(),
            "new_dirs": set(),
        }
        if metadata_enabled:
            metadata_path = Path(abs_dataset) / metadata_name
            stored = _load_processed_dirs(metadata_path, reset=reset_metadata)
            pruned = _prune_missing_dirs(abs_dataset, stored)
            ctx.update(
                {
                    "metadata_path": metadata_path,
                    "stored_dirs": stored,
                    "processed_dirs": pruned,
                }
            )
        dataset_contexts.append(ctx)

    if not dataset_contexts:
        return []

    skip_map = (
        {ctx["abs_path"]: ctx["processed_dirs"] for ctx in dataset_contexts}
        if metadata_enabled
        else None
    )
    estimated_total = _estimate_total_frames(
        [ctx["abs_path"] for ctx in dataset_contexts],
        normalized_exts,
        skip_map=skip_map,
    )

    processed = 0
    all_entries = []
    for ctx in dataset_contexts:
        abs_dataset = ctx["abs_path"]
        processed_dirs = ctx["processed_dirs"]
        metadata_path = ctx["metadata_path"]
        dataset_start = len(all_entries)
        for root, dirs, files in os.walk(abs_dataset):
            if metadata_enabled and processed_dirs and root == abs_dataset:
                dirs[:] = [d for d in dirs if d not in processed_dirs]
            for file in files:
                if not file.lower().endswith(normalized_exts):
                    continue
                img_path = os.path.join(root, file)
                json_path = os.path.splitext(img_path)[0] + ".json"
                processed += 1
                _print_progress(processed, estimated_total)
                if os.path.isfile(json_path):
                    all_entries.append((img_path, file, json_path))
                    if metadata_enabled:
                        rel_path = os.path.relpath(img_path, abs_dataset)
                        parts = rel_path.split(os.sep)
                        if len(parts) > 1:
                            ctx["new_dirs"].add(parts[0])
        dataset_count = len(all_entries) - dataset_start
        print(f"\n{abs_dataset}: {dataset_count} image(s) with annotations found.")
        if metadata_enabled and metadata_path:
            updated_dirs = ctx["processed_dirs"] | ctx["new_dirs"]
            if updated_dirs != ctx["stored_dirs"]:
                _write_processed_dirs(metadata_path, updated_dirs)
    print()
    return all_entries


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate local DOPE training data and optionally write a manifest "
            "file that train.py can consume via --training_manifest."
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
        default=None,
        help=(
            "Where to write the manifest JSON (default: training_manifest.json "
            "inside the dataset directory)."
        ),
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
    parser.add_argument(
        "--metadata-name",
        default=DEFAULT_METADATA_NAME,
        help=(
            "Filename created in each dataset root to store processed subfolders "
            f"(default: {DEFAULT_METADATA_NAME})."
        ),
    )
    parser.add_argument(
        "--disable-metadata",
        action="store_true",
        help="Skip metadata tracking so every subfolder is rescanned.",
    )
    parser.add_argument(
        "--reset-metadata",
        action="store_true",
        help="Ignore existing metadata files and rewrite them after this run.",
    )
    args = parser.parse_args()

    metadata_name = None if args.disable_metadata else args.metadata_name
    entries = _collect_entries(
        args.data,
        extensions=args.extensions,
        metadata_name=metadata_name,
        reset_metadata=args.reset_metadata,
    )
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

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        dataset_root = Path(args.data[0]).expanduser().resolve()
        output_path = dataset_root / "training_manifest.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2))
    print(
        f"Saved manifest with {len(manifest)} entries to '{output_path}'. "
        "Pass this path to train.py via --training_manifest to skip the directory scan."
    )


if __name__ == "__main__":
    main()
