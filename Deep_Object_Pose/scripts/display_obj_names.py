#!/usr/bin/env python3
from pathlib import Path

OBJ_PATH = Path("./data/models/Antenna/antenna.obj")

def extract_parts(obj_path: Path):
    parts = set()
    if not obj_path.is_file():
        raise FileNotFoundError(f"OBJ file '{obj_path}' not found.")
    with obj_path.open() as fh:
        for line in fh:
            lower = line.strip().lower()
            if lower.startswith("o ") or lower.startswith("g ") or lower.startswith("usemtl "):
                parts.add(line.strip())
    return sorted(parts)

if __name__ == "__main__":
    for entry in extract_parts(OBJ_PATH):
        print(entry)
