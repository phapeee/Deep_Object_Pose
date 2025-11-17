#!/usr/bin/env python3

import argparse
import json
import multiprocessing
import os
from pathlib import Path
from queue import Queue
import re
import subprocess
import sys
import threading


parser = argparse.ArgumentParser()
## Parameters for this script
parser.add_argument(
    '--nb_runs',
    default=1,
    type=int,
    help='Number of times the datagen script is run. Each time it is run, a new set of '
    'distractors is selected.'
)
parser.add_argument(
    '--nb_workers',
    default=0,
    type=int,
    help='Number of parallel blenderproc workers to run.  The default of 0 will create '
    'one worker for every CPU core'
)
parser.add_argument(
    '--config',
    default=None,
    help='Path to sence_config.json containing datagen_defaults used as CLI arguments.'
)


opt, unknown = parser.parse_known_args()

num_workers = min(opt.nb_workers, multiprocessing.cpu_count())
if num_workers == 0:
    num_workers = multiprocessing.cpu_count()

# set the folder in which the generation script is located
rerun_folder = os.path.abspath(os.path.dirname(__file__))

progress = {}
progress_lock = threading.Lock()


def _default_config_path():
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "config" / "sence_config.json"
    if candidate.is_file():
        return candidate
    return None


def load_datagen_defaults(config_path=None):
    candidate = Path(config_path) if config_path else _default_config_path()
    if not candidate or not candidate.is_file():
        return None
    try:
        data = json.loads(candidate.read_text())
    except Exception:
        return None
    defaults = data.get("datagen_defaults")
    if not isinstance(defaults, dict):
        return None
    cleaned = {}
    for key, value in defaults.items():
        cleaned[str(key)] = value
    return cleaned


def existing_cli_flags(args_list):
    flags = set()
    for token in args_list:
        if token.startswith("--"):
            flag = token.split("=", 1)[0]
            flags.add(flag)
    return flags


def convert_defaults_to_args(defaults, skip_flags):
    if not defaults:
        return []
    result = []
    for key, value in defaults.items():
        flag = f"--{key}"
        if flag in skip_flags:
            continue
        if isinstance(value, bool):
            if value:
                result.append(flag)
            continue
        if isinstance(value, list):
            if not value:
                continue
            result.append(flag)
            result.extend(str(item) for item in value)
            continue
        result.append(flag)
        result.append(str(value))
    return result


def format_progress():
    with progress_lock:
        if not progress:
            return ""
        parts = [f"R{run_id}: {progress[run_id]}" for run_id in sorted(progress)]
        return " | ".join(parts)


def render_progress_line():
    status_line = format_progress()
    if status_line:
        print(f"\r{status_line}", end='', flush=True)


def stream_output(proc, run_id):
    for line in proc.stdout:
        stripped = line.rstrip()
        if "Finished rendering after" in stripped:
            continue

        match = re.match(r"Run (\d+)(?: [^:]+)?: (\d+)/(\d+)", stripped)
        if match:
            _, current, total = match.groups()
            with progress_lock:
                progress[run_id] = f"{current}/{total}"
            render_progress_line()
            continue

        print()
        print(f"[run {run_id}] {stripped}")


Q = Queue(maxsize = num_workers)
active_processes = []

config_defaults = load_datagen_defaults(opt.config)
config_nb_runs = None
if config_defaults and "nb_runs" in config_defaults:
    try:
        config_nb_runs = int(config_defaults.pop("nb_runs"))
    except (TypeError, ValueError):
        config_nb_runs = None

default_nb_runs = parser.get_default('nb_runs')
if opt.nb_runs == default_nb_runs and config_nb_runs is not None:
    amount_of_runs = config_nb_runs
else:
    amount_of_runs = opt.nb_runs

def wait_for_process():
    run_identifier, proc, thread = Q.get()
    proc.wait()
    thread.join()
    with progress_lock:
        progress.pop(run_identifier, None)
    active_processes.remove((run_identifier, proc, thread))
    render_progress_line()

try:
    for run_id in range(amount_of_runs):
        if Q.full():
            wait_for_process()

        # execute one BlenderProc run
        cmd = ["blenderproc", "run", os.path.join(rerun_folder, "generate_training_data.py")]
        skip_flags = existing_cli_flags(unknown)
        cmd.extend(convert_defaults_to_args(config_defaults, skip_flags))
        cmd.extend(unknown)
        cmd.extend(['--run_id', str(run_id)])
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        t = threading.Thread(target=stream_output, args=(p, run_id), daemon=True)
        t.start()
        Q.put((run_id, p, t))
        active_processes.append((run_id, p, t))

    # ensure all workers finish before exiting
    while not Q.empty():
        wait_for_process()
except KeyboardInterrupt:
    print("\nKeyboard interrupt received, terminating BlenderProc runs and clearing GPU memory...", flush=True)
    for _, proc, _ in active_processes:
        if proc.poll() is None:
            proc.terminate()
    for run_identifier, proc, thread in active_processes:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        thread.join()
        with progress_lock:
            progress.pop(run_identifier, None)
        render_progress_line()
    raise
finally:
    print()
