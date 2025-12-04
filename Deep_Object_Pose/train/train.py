#!/usr/bin/python3

"""
Example usage:

 torchrun --nproc_per_node=1 train.py --data ../sample_data/ --object cracker
"""


import argparse
import copy
import datetime
import json
import os
from pathlib import Path
from queue import Queue
import random
import time
import warnings
warnings.filterwarnings("ignore")

try:
    import configparser as configparser
except ImportError:
    import ConfigParser as configparser

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMON_DIR = os.path.join(ROOT_DIR, "common")
if COMMON_DIR not in sys.path:
    sys.path.insert(1, COMMON_DIR)

OUTPUT_ROOT_DIR = Path(ROOT_DIR) / "output"

from models import *
from utils import *

DEFAULT_KEYPOINT_ACCURACY_TOLERANCE = 5.0
DEFAULT_MANIFEST_FILENAME = "training_manifest.json"
DEFAULT_VAL_LOG_NAME = "validation_log.csv"

class _AutoBatchRestart(Exception):
    def __init__(self, batch_size):
        self.batch_size = batch_size

try:
    from validate_training_data import (
        _collect_entries as _collect_training_entries,
        DEFAULT_METADATA_NAME as _TRAINING_METADATA_NAME,
    )
except Exception:
    _collect_training_entries = None
    _TRAINING_METADATA_NAME = None


class _BatchSizeController:
    def __init__(self, initial):
        self.value = max(1, int(initial))

    def get(self):
        return max(1, int(self.value))

    def set(self, new_value):
        self.value = max(1, int(new_value))


class _DynamicBatchSampler(torch.utils.data.Sampler):
    def __init__(self, base_sampler, controller, drop_last=False):
        self.base_sampler = base_sampler
        self.controller = controller
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.base_sampler:
            batch.append(idx)
            current_size = self.controller.get()
            if len(batch) >= current_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        total = len(self.base_sampler)
        batch_size = self.controller.get()
        if self.drop_last:
            return total // batch_size
        return (total + batch_size - 1) // batch_size


def _normalize_manifest_payload(payload):
    normalized = []
    for entry in payload or []:
        img_path = entry.get("image_path")
        json_path = entry.get("json_path")
        if not img_path or not json_path:
            continue
        image_name = entry.get("image_name") or os.path.basename(img_path)
        normalized.append((img_path, image_name, json_path))
    return normalized


def _load_manifest_file(manifest_path):
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text())
    except Exception as exc:
        print(f"[data manifest] Failed to parse '{manifest_path}': {exc}. Regenerating.")
        return None
    entries = _normalize_manifest_payload(payload)
    if not entries:
        print(f"[data manifest] '{manifest_path}' is empty. Regenerating.")
        return None
    print(f"[data manifest] Loaded {len(entries)} entries from '{manifest_path}'.")
    return entries


def _write_manifest_file(manifest_path, entries):
    manifest_path = Path(manifest_path)
    payload = [
        {
            "image_path": img_path,
            "json_path": json_path,
            "image_name": image_name,
        }
        for img_path, image_name, json_path in entries
    ]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2))


def _archive_output_directory(source_dir, archive_path):
    source_dir = Path(source_dir)
    archive_path = Path(archive_path)
    if not source_dir.exists():
        print(f"[gdrive] Output directory '{source_dir}' not found; skipping archive.")
        return None
def _prepare_training_manifest_entries(
    dataset_paths, manifest_name=DEFAULT_MANIFEST_FILENAME, extensions=None
):
    if not dataset_paths:
        return None
    extensions = extensions or ["png"]

    combined_entries = []
    validation_available = _collect_training_entries is not None
    for dataset in dataset_paths:
        abs_dataset = os.path.abspath(dataset)
        if not os.path.isdir(abs_dataset):
            print(f"[data manifest] Skipping '{abs_dataset}' (directory not found).")
            continue
        manifest_path = os.path.join(abs_dataset, manifest_name)
        entries = _load_manifest_file(manifest_path)
        if entries is None:
            if not validation_available:
                print(
                    f"[data manifest] '{manifest_path}' missing but validation helper unavailable. "
                    "Falling back to directory scan for this dataset."
                )
                return None
            print(f"[data manifest] Creating '{manifest_path}' via validation scan...")
            entries = _collect_training_entries(
                [abs_dataset],
                extensions=extensions,
                metadata_name=_TRAINING_METADATA_NAME,
                reset_metadata=False,
            )
            if not entries:
                print(f"[data manifest] No annotated samples discovered under '{abs_dataset}'.")
                continue
            _write_manifest_file(manifest_path, entries)
            print(f"[data manifest] Saved {len(entries)} entries to '{manifest_path}'.")
        combined_entries.extend(entries)

    if not combined_entries:
        return None

    deduped_entries = []
    seen = set()
    for img_path, img_name, json_path in combined_entries:
        key = (img_path, json_path)
        if key in seen:
            continue
        seen.add(key)
        deduped_entries.append((img_path, img_name, json_path))
    return deduped_entries


def _reset_peak_memory_stats(device):
    if hasattr(torch.cuda, "reset_peak_memory_stats"):
        torch.cuda.reset_peak_memory_stats(device)
    else:
        torch.cuda.reset_max_memory_allocated(device)
        if hasattr(torch.cuda, "reset_max_memory_cached"):
            torch.cuda.reset_max_memory_cached(device)


def _get_free_gpu_memory(device):
    if hasattr(torch.cuda, "mem_get_info"):
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    else:
        props = torch.cuda.get_device_properties(device)
        total_bytes = props.total_memory
        reserved = torch.cuda.memory_reserved(device)
        allocated = torch.cuda.memory_allocated(device)
        free_bytes = total_bytes - max(reserved, allocated)
    return free_bytes, total_bytes


def _estimate_batch_from_peak_usage(
    peak_memory_bytes,
    current_batch,
    total_memory_bytes,
    headroom_frac=0.1,
    max_batch_size=None,
):
    if peak_memory_bytes <= 0 or current_batch <= 0:
        return current_batch, 0.0
    target_bytes = float(total_memory_bytes) - float(headroom_frac)
    target_bytes = max(target_bytes, 1.0)
    peak_memory_bytes = max(float(peak_memory_bytes), 1.0)
    ratio = target_bytes / peak_memory_bytes
    # avoid extreme jumps to keep training stable
    ratio = max(0.5, min(ratio, 2.0))
    estimated_batch = max(1, int(current_batch * ratio))
    if max_batch_size is not None:
        estimated_batch = min(estimated_batch, max(1, int(max_batch_size)))
    per_sample_usage = peak_memory_bytes / float(current_batch)
    return estimated_batch, per_sample_usage


def _build_training_loader(dataset, batch_size, workers, controller=None):
    if controller is None:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )
    base_sampler = torch.utils.data.RandomSampler(dataset)
    batch_sampler = _DynamicBatchSampler(base_sampler, controller, drop_last=False)
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=workers,
        pin_memory=True,
    )


def _maybe_update_auto_batch(auto_ctx, local_rank, force=False):
    if not auto_ctx:
        return
    if auto_ctx.get("auto_disabled"):
        return
    cooldown = auto_ctx.get("cooldown_counter", 0)
    if cooldown > 0 and not force:
        auto_ctx["cooldown_counter"] = cooldown - 1
        return
    controller = auto_ctx["controller"]
    device = auto_ctx["device"]
    current_size = controller.get()
    torch.cuda.synchronize(device)
    sample = max(
        torch.cuda.memory_allocated(device),
        torch.cuda.memory_reserved(device),
    )
    if auto_ctx.get("debug_samples"):
        print(f"[auto-batch] VRAM sample: {sample / (1024 ** 3):.2f} GB (batch {current_size})")
    remaining = auto_ctx.get("samples_remaining")
    ema = auto_ctx.get("steady_state_bytes")
    if ema is None or remaining is None:
        ema = float(sample)
        remaining = auto_ctx.get("samples_to_collect", 0)
    else:
        ema = 0.9 * ema + 0.1 * float(sample)
    auto_ctx["steady_state_bytes"] = ema
    if remaining > 0:
        auto_ctx["samples_remaining"] = remaining - 1
        return
    observed_peak = max(auto_ctx.get("last_peak", 0.0), ema, float(sample))
    suggested_bs, per_sample = _estimate_batch_from_peak_usage(
        ema,
        current_size,
        auto_ctx["total_memory"],
        auto_ctx["headroom"],
        auto_ctx.get("max_batch_size"),
    )
    suggested_bs = max(1, int(suggested_bs))
    if suggested_bs == current_size:
        stable_count = auto_ctx.get("stable_counter", 0) + 1
        auto_ctx["stable_counter"] = stable_count
        pending = auto_ctx.get("pending_verification", False)
        stable_limit = auto_ctx.get("stable_limit", 0)
        if pending or (stable_limit and stable_count >= stable_limit):
            auto_ctx["auto_disabled"] = True
            if auto_ctx.get("debug_samples"):
                print(
                    f"[auto-batch] Batch size stabilized at {current_size}; disabling further sampling."
                )
            auto_ctx["pending_verification"] = False
            auto_ctx["request_restart"] = True
            auto_ctx["restart_batch_size"] = current_size
            return
        auto_ctx["samples_remaining"] = auto_ctx.get("samples_to_collect", 0)
        auto_ctx["cooldown_counter"] = auto_ctx.get("cooldown_steps", 0)
        return

    prev_size = auto_ctx.get("last_batch_size")
    if (
        prev_size
        and prev_size == suggested_bs + 1
        and current_size == suggested_bs - 1
    ):
        controller.set(prev_size)
        auto_ctx["opt_ref"].batchsize = prev_size
        auto_ctx["auto_disabled"] = True
        auto_ctx["request_restart"] = True
        auto_ctx["restart_batch_size"] = prev_size
        if auto_ctx.get("debug_samples"):
            print(
                f"[auto-batch] Detected oscillation around {suggested_bs}; "
                f"locking batch size at {prev_size}."
            )
        return
    auto_ctx["stable_counter"] = 0
    auto_ctx["last_batch_size"] = suggested_bs
    exchange = auto_ctx["exchange_tensor"]
    exchange[0] = current_size
    if local_rank == 0:
        exchange[0] = suggested_bs
    if torch.distributed.is_initialized():
        torch.distributed.broadcast(exchange, src=0)
    new_size = int(exchange.item())
    new_size = max(1, new_size)
    if new_size == current_size:
        return
    controller.set(new_size)
    auto_ctx["last_batch_size"] = new_size
    auto_ctx["last_peak"] = observed_peak
    auto_ctx["steady_state_bytes"] = None
    auto_ctx["samples_remaining"] = 1
    auto_ctx["pending_verification"] = True
    auto_ctx["auto_disabled"] = False
    auto_ctx["last_per_sample"] = per_sample
    if "opt_ref" in auto_ctx and auto_ctx["opt_ref"] is not None:
        auto_ctx["opt_ref"].batchsize = new_size
    auto_ctx["cooldown_counter"] = auto_ctx.get("cooldown_steps", 0)
    if local_rank == 0:
        direction = "increasing" if new_size > current_size else "decreasing"
        print(
            f"[auto-batch] {direction.capitalize()} batch size to {new_size} "
            f"(previously {current_size}). Peak {observed_peak / (1024**3):.2f} GB "
            f"~{per_sample / (1024**2):.1f} MB/sample."
        )
    _reset_peak_memory_stats(device)


def _keypoint_accuracy(output_belief, target_belief, tolerance_px):
    if not output_belief:
        return 0.0
    try:
        pred_maps = output_belief[-1].detach()
    except Exception:
        return 0.0
    target_maps = target_belief.detach()
    if pred_maps.shape != target_maps.shape:
        min_channels = min(pred_maps.shape[1], target_maps.shape[1])
        pred_maps = pred_maps[:, :min_channels]
        target_maps = target_maps[:, :min_channels]
    b, c, h, w = pred_maps.shape
    if b == 0 or c == 0:
        return 0.0
    pred_flat = pred_maps.reshape(b, c, -1)
    target_flat = target_maps.reshape(b, c, -1)
    pred_idx = pred_flat.argmax(dim=2)
    target_idx = target_flat.argmax(dim=2)
    pred_y = pred_idx // w
    pred_x = pred_idx % w
    target_y = target_idx // w
    target_x = target_idx % w
    dist = torch.sqrt(
        (pred_y - target_y).float().pow(2) + (pred_x - target_x).float().pow(2)
    )
    return dist.le(float(tolerance_px)).float().mean().item()


def _runnetwork(
    net,
    optimizer,
    local_rank,
    epoch,
    train_loader,
    writer=None,
    accuracy_tolerance_px=DEFAULT_KEYPOINT_ACCURACY_TOLERANCE,
    log_interval=None,
    auto_batch_ctx=None,
):
    epoch_start = time.time()
    num_batches = len(train_loader)
    batch_log_interval = None
    if log_interval is not None:
        try:
            batch_log_interval = max(1, int(log_interval))
        except Exception:
            batch_log_interval = None
    loss_avg_to_log = {}
    loss_avg_to_log["loss"] = []
    loss_avg_to_log["loss_affinities"] = []
    loss_avg_to_log["loss_belief"] = []
    loss_avg_to_log["loss_class"] = []
    loss_avg_to_log["accuracy"] = []
    samples_seen = 0
    for batch_idx, targets in enumerate(train_loader):
        optimizer.zero_grad()

        data = Variable(targets["img"].cuda())
        target_belief = Variable(targets["beliefs"].cuda())
        target_affinities = Variable(targets["affinities"].cuda())

        output_belief, output_aff = net(data)

        loss = None

        loss_belief = torch.tensor(0).float().cuda()
        loss_affinities = torch.tensor(0).float().cuda()
        loss_class = torch.tensor(0).float().cuda()

        for stage in range(len(output_aff)):  # output, each belief map layers.
            loss_affinities += (
                (output_aff[stage] - target_affinities)
                * (output_aff[stage] - target_affinities)
            ).mean()

            loss_belief += (
                (output_belief[stage] - target_belief)
                * (output_belief[stage] - target_belief)
            ).mean()

        loss = loss_affinities + loss_belief

        batch_accuracy = _keypoint_accuracy(
            output_belief, target_belief, accuracy_tolerance_px
        )

        if batch_idx == 0:
            post = "train"

            if writer is not None and local_rank == 0:
                for i_output in range(1):

                    # input images
                    writer.add_image(
                        f"{post}_input_{i_output}",
                        targets["img_original"][i_output],
                        epoch,
                        dataformats="CWH",
                    )

                    # belief maps gt
                    imgs = VisualizeBeliefMap(target_belief[i_output])
                    imgs[imgs == float('inf')] = 0
                    img, grid = save_image(
                        imgs, "belief_maps_gt.png", mean=0, std=1, nrow=3, save=False
                    )
                    writer.add_image(
                        f"{post}_belief_ground_truth_{i_output}",
                        grid,
                        epoch,
                        dataformats="CWH",
                    )

                    # belief maps guess
                    imgs = VisualizeBeliefMap(output_belief[-1][i_output])
                    imgs[imgs == float('inf')] = 0
                    img, grid = save_image(
                        imgs, "belief_maps.png", mean=0, std=1, nrow=3, save=False
                    )
                    writer.add_image(
                        f"{post}_belief_guess_{i_output}",
                        grid,
                        epoch,
                        dataformats="CWH",
                    )


        loss.backward()

        optimizer.step()

        # log the loss
        loss_avg_to_log["loss"].append(loss.item())
        loss_avg_to_log["loss_class"].append(loss_class.item())
        loss_avg_to_log["loss_affinities"].append(loss_affinities.item())
        loss_avg_to_log["loss_belief"].append(loss_belief.item())
        loss_avg_to_log["accuracy"].append(batch_accuracy)

        samples_seen += len(data)
        processed = min(samples_seen, len(train_loader.dataset))
        percent = 100.0 * processed / len(train_loader.dataset)
        print(
            f"Train Epoch: {epoch} [{processed}/{len(train_loader.dataset)} ({percent:.0f}%)] "
            f"\tLoss: {loss.item():.15f} \tAccuracy: {batch_accuracy * 100:6.2f}%"
            f"\tLocal Rank: {local_rank}"
        )
        should_log = (
            batch_log_interval is not None
            and (batch_idx % batch_log_interval == 0 or batch_idx + 1 == num_batches)
        )
        if (
            writer is not None
            and local_rank == 0
            and should_log
        ):
            elapsed = max(time.time() - epoch_start, 1e-9)
            samples_per_sec = samples_seen / elapsed
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar("loss/batch_loss", loss.item(), global_step)
            writer.add_scalar("loss/batch_affinities", loss_affinities.item(), global_step)
            writer.add_scalar("loss/batch_belief", loss_belief.item(), global_step)
            writer.add_scalar(
                "metrics/batch_accuracy",
                batch_accuracy,
                global_step,
            )
            writer.add_scalar(
                "metrics/samples_per_sec_mid_epoch",
                samples_per_sec,
                global_step,
            )
        if auto_batch_ctx:
            _maybe_update_auto_batch(auto_batch_ctx, local_rank)

    # log the loss values
    if writer is not None and local_rank == 0:
        writer.add_scalar(
            "loss/train_loss", np.mean(loss_avg_to_log["loss"]), epoch
        )
        writer.add_scalar(
            "loss/train_cls", np.mean(loss_avg_to_log["loss_class"]), epoch
        )
        writer.add_scalar(
            "loss/train_aff", np.mean(loss_avg_to_log["loss_affinities"]), epoch
        )
        writer.add_scalar(
            "loss/train_bel", np.mean(loss_avg_to_log["loss_belief"]), epoch
        )
        writer.add_scalar(
            "metrics/train_accuracy",
            np.mean(loss_avg_to_log["accuracy"]),
            epoch,
        )
    if auto_batch_ctx:
        _maybe_update_auto_batch(auto_batch_ctx, local_rank, force=True)

    epoch_duration = max(time.time() - epoch_start, 1e-9)
    return {
        "samples_seen": samples_seen,
        "duration": epoch_duration,
    }


def _run_validation(
    net,
    local_rank,
    epoch,
    val_loader,
    writer=None,
    accuracy_tolerance_px=DEFAULT_KEYPOINT_ACCURACY_TOLERANCE,
    max_batches=None,
    log_path=None,
):
    if val_loader is None or len(val_loader) == 0:
        return None
    start_time = time.time()
    was_training = net.training
    net.eval()
    limit_batches = max(0, int(max_batches or 0))
    loss_values = {"loss": [], "loss_aff": [], "loss_bel": [], "accuracy": []}
    samples_seen = 0
    processed_batches = 0
    with torch.no_grad():
        for batch_idx, targets in enumerate(val_loader):
            if limit_batches and batch_idx >= limit_batches:
                break
            data = targets["img"].cuda(non_blocking=True)
            target_belief = targets["beliefs"].cuda(non_blocking=True)
            target_affinities = targets["affinities"].cuda(non_blocking=True)

            output_belief, output_aff = net(data)

            loss_affinities = torch.tensor(0.0, device=data.device)
            loss_belief = torch.tensor(0.0, device=data.device)
            for stage in range(len(output_aff)):
                loss_affinities += (
                    (output_aff[stage] - target_affinities)
                    * (output_aff[stage] - target_affinities)
                ).mean()
                loss_belief += (
                    (output_belief[stage] - target_belief)
                    * (output_belief[stage] - target_belief)
                ).mean()
            total_loss = loss_affinities + loss_belief

            batch_accuracy = _keypoint_accuracy(
                output_belief, target_belief, accuracy_tolerance_px
            )

            loss_values["loss"].append(total_loss.item())
            loss_values["loss_aff"].append(loss_affinities.item())
            loss_values["loss_bel"].append(loss_belief.item())
            loss_values["accuracy"].append(batch_accuracy)

            processed_batches += 1
            samples_seen += len(data)

    if was_training:
        net.train()

    if processed_batches == 0:
        return None

    avg_loss = float(np.mean(loss_values["loss"]))
    avg_aff = float(np.mean(loss_values["loss_aff"]))
    avg_bel = float(np.mean(loss_values["loss_bel"]))
    avg_acc = float(np.mean(loss_values["accuracy"]))

    duration = max(time.time() - start_time, 1e-9)
    percent = avg_acc * 100.0
    print(
        f"[validation] Epoch {epoch}: loss={avg_loss:.6f} "
        f"(aff {avg_aff:.6f}, belief {avg_bel:.6f}) "
        f"accuracy={percent:6.2f}% "
        f"({samples_seen} samples, {processed_batches} batches, {duration:.2f}s)"
    )

    if writer is not None and local_rank == 0:
        writer.add_scalar("loss/val_loss", avg_loss, epoch)
        writer.add_scalar("loss/val_aff", avg_aff, epoch)
        writer.add_scalar("loss/val_bel", avg_bel, epoch)
        writer.add_scalar("metrics/val_accuracy", avg_acc, epoch)

    if log_path is not None and local_rank == 0:
        log_exists = Path(log_path).exists()
        with open(log_path, "a") as log_file:
            if not log_exists:
                log_file.write("epoch,loss,loss_aff,loss_bel,accuracy,samples,batches,duration_sec\n")
            log_file.write(
                f"{epoch},{avg_loss:.10f},{avg_aff:.10f},{avg_bel:.10f},"
                f"{avg_acc:.10f},{samples_seen},{processed_batches},{duration:.4f}\n"
            )

    return {
        "loss": avg_loss,
        "loss_aff": avg_aff,
        "loss_bel": avg_bel,
        "accuracy": avg_acc,
        "samples_seen": samples_seen,
        "batches": processed_batches,
        "duration": duration,
    }


def main(opt):
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.gradcheck = False
    torch.backends.cudnn.benchmark = True

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_LOCALID" in os.environ:
        local_rank = int(os.environ["SLURM_LOCALID"])
    else:
        local_rank = getattr(opt, "local_rank", 0)

    # Validate Arguments
    if opt.use_s3 and (opt.train_buckets is None or opt.endpoint is None):
        raise ValueError(
            "--train_buckets and --endpoint must be specified if training with data from s3 bucket."
        )

    if not opt.use_s3 and opt.data is None:
        raise ValueError("--data field must be specified.")

    os.makedirs(opt.outf, exist_ok=True)
    val_log_path = os.path.join(opt.outf, DEFAULT_VAL_LOG_NAME)

    random_seed = random.randint(1, 10000)
    if opt.manualseed is not None:
        random_seed = opt.manualseed

    # Save run parameters in a file
    with open(opt.outf + "/header.txt", "w") as file:
        file.write(str(opt) + "\n")
        file.write("seed: " + str(random_seed) + "\n")

    writer = None
    if local_rank == 0:
        writer = SummaryWriter(opt.outf + "/runs/")

    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="NCCL", init_method="env://")


    # Data Augmentation
    transform = transforms.Compose([
        transforms.Resize(opt.imagesize),
        transforms.ToTensor()
    ])

    # Load Model
    net = DopeNetwork()
    output_size = 50
    opt.sigma = 0.5

    # Convert object names to lower-case for comparison later
    for idx in range(len(opt.object)):
        opt.object[idx] = opt.object[idx].lower()

    manifest_entries = None
    dataset_paths = list(opt.data or [])
    if not opt.use_s3 and dataset_paths:
        manifest_entries = _prepare_training_manifest_entries(dataset_paths)
        if manifest_entries is not None:
            dataset_paths = []

    training_dataset = CleanVisiiDopeLoader(
        dataset_paths,
        sigma=opt.sigma,
        output_size=output_size,
        objects=opt.object,
        use_s3=opt.use_s3,
        buckets=opt.train_buckets,
        endpoint_url=opt.endpoint,
    )
    if manifest_entries is not None:
        training_dataset.imgs = manifest_entries
        print(f"Number of Training Images (from manifest): {len(training_dataset.imgs)}")

    batch_controller = None
    if getattr(opt, "auto_batchsize", False):
        batch_controller = _BatchSizeController(opt.batchsize)

    training_data = _build_training_loader(
        training_dataset, opt.batchsize, opt.workers, controller=batch_controller
    )
    if getattr(opt, "auto_batchsize", False) and local_rank == 0:
        print(
            "[auto-batch] Dynamic VRAM tracking enabled. "
            f"Starting batch size {opt.batchsize} with headroom "
            f"{getattr(opt, 'auto_batch_headroom', 0.1):.2f}."
        )

    if not training_data is None:
        print("training data: {} batches".format(len(training_data)))

        print("Loading Model...")
        net = torch.nn.parallel.DistributedDataParallel(
            net.cuda(),
            device_ids=[local_rank],
            output_device=local_rank
        )

    validation_loader = None
    val_paths = list(getattr(opt, "val_data", None) or [])
    if val_paths:
        val_manifest_entries = None
        if not opt.use_s3:
            val_manifest_entries = _prepare_training_manifest_entries(val_paths)
            if val_manifest_entries is not None:
                if local_rank == 0:
                    print(
                        f"[validation manifest] Loaded {len(val_manifest_entries)} entries "
                        "from validation manifest(s)."
                    )
                val_paths = []
        validation_dataset = CleanVisiiDopeLoader(
            val_paths,
            sigma=opt.sigma,
            output_size=output_size,
            objects=opt.object,
            use_s3=opt.use_s3,
            buckets=opt.train_buckets,
            endpoint_url=opt.endpoint,
        )
        if val_manifest_entries is not None:
            validation_dataset.imgs = val_manifest_entries
        if len(validation_dataset) == 0:
            if local_rank == 0:
                print("[validation] No samples found in validation dataset; disabling validation.")
        else:
            val_batchsize = getattr(opt, "val_batchsize", None) or opt.batchsize
            val_workers = getattr(opt, "val_workers", None)
            if val_workers is None:
                val_workers = opt.workers
            validation_loader = torch.utils.data.DataLoader(
                validation_dataset,
                batch_size=val_batchsize,
                shuffle=False,
                num_workers=val_workers,
                pin_memory=True,
            )
            if local_rank == 0:
                print(
                    f"[validation] dataset contains {len(validation_dataset)} samples "
                    f"({len(validation_loader)} batches)."
                )
    elif getattr(opt, "validate_every", 0):
        if local_rank == 0:
            print(
                "[validation] Warning: --validate_every is set but no validation data was provided. "
                "Validation will be skipped."
            )

    # Load any previous checkpoint (i.e. current job is a follow-up job)
    if opt.net_path is not None:
        net.load_state_dict(torch.load(opt.net_path))

    parameters = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.Adam(parameters, lr=opt.lr)

    print("ready to train!")
    start_time = datetime.datetime.now()
    runtime_limit_seconds = 0.0
    runtime_limit_source = "max_runtime_seconds"
    configured_threshold = getattr(opt, "time_threshold_seconds", None)
    if configured_threshold is not None and configured_threshold > 0:
        runtime_limit_seconds = float(configured_threshold)
        runtime_limit_source = "time_threshold_seconds"
    else:
        runtime_limit_seconds = float(getattr(opt, "max_runtime_seconds", 0) or 0)
    runtime_limit_seconds = max(0.0, runtime_limit_seconds)
    runtime_timer_start = time.monotonic()
    ready_to_exit = False
    opt.ready_to_exit = False
    if runtime_limit_seconds > 0 and local_rank == 0:
        limit_minutes = runtime_limit_seconds / 60.0
        print(
            f"[runtime limit] Will request shutdown after ~{runtime_limit_seconds:.0f}s "
            f"({limit_minutes:.2f} min) once the current epoch completes (config: {runtime_limit_source})."
        )
    print("start:", start_time.strftime("%m/%d/%Y, %H:%M:%S"))

    ckpt_q = None
    if opt.nb_checkpoints > 0:
        ckpt_q = Queue(maxsize=opt.nb_checkpoints)

    start_epoch = 0
    if opt.net_path is not None:
        # We started with a saved checkpoint, we start numbering checkpoints
        # after the loaded one
        try:
            start_epoch = int(os.path.splitext(os.path.basename(opt.net_path).split('_')[-1])[0]) + 1
        except:
            start_epoch = 1
        print(f"Starting at epoch {start_epoch}")

    _, total_gpu_mem = _get_free_gpu_memory(device)
    auto_batch_ctx = None
    if getattr(opt, "auto_batchsize", False):
        auto_batch_ctx = {
            "controller": batch_controller,
            "device": device,
            "total_memory": total_gpu_mem,
            "headroom": getattr(opt, "auto_batch_headroom", 0.1),
            "exchange_tensor": torch.tensor([opt.batchsize], device=device, dtype=torch.int32),
            "opt_ref": opt,
            "max_batch_size": getattr(opt, "auto_batch_max", 128),
            "cooldown_steps": max(0, int(getattr(opt, "auto_batch_cooldown", 0))),
            "cooldown_counter": max(0, int(getattr(opt, "auto_batch_cooldown", 0))),
            "samples_to_collect": max(1, int(getattr(opt, "auto_batch_samples", 5))),
            "samples_remaining": max(1, int(getattr(opt, "auto_batch_samples", 5))),
            "debug_samples": getattr(opt, "auto_batch_debug", False),
            "stable_limit": max(0, int(getattr(opt, "auto_batch_stable_checks", 0))),
            "stable_counter": 0,
        }

    best_val_accuracy = float("-inf")
    best_val_epoch = None
    best_val_path = os.path.join(opt.outf, "best_val_net.pth")

    try:
        net.train()
        for epoch in range(start_epoch, opt.epochs + 1):
            if getattr(opt, "auto_batchsize", False):
                torch.cuda.empty_cache()
                _reset_peak_memory_stats(device)

            perf_stats = _runnetwork(
                net,
                optimizer,
                local_rank,
                epoch,
                training_data,
                writer,
                accuracy_tolerance_px=opt.accuracy_px_tolerance,
                log_interval=getattr(opt, "loginterval", None),
                auto_batch_ctx=auto_batch_ctx,
            )

            if perf_stats and local_rank == 0:
                samples = perf_stats.get("samples_seen", 0)
                duration = max(perf_stats.get("duration", 0.0), 1e-9)
                samples_per_sec = samples / duration
                print(
                    f"[throughput] Epoch {epoch}: {samples_per_sec:.2f} samples/sec "
                    f"({samples} samples in {duration:.2f}s)"
                )
                if writer is not None:
                    writer.add_scalar("metrics/samples_per_sec", samples_per_sec, epoch)

            elapsed_since_start = time.monotonic() - runtime_timer_start
            opt.elapsed_runtime_seconds = elapsed_since_start
            if local_rank == 0:
                if runtime_limit_seconds > 0:
                    remaining = max(0.0, runtime_limit_seconds - elapsed_since_start)
                    print(
                        "[runtime] Elapsed "
                        f"{elapsed_since_start:.2f}s since start "
                        f"(threshold {runtime_limit_seconds:.2f}s, "
                        f"{remaining:.2f}s remaining)."
                    )
            else:
                print(
                    f"[runtime] Elapsed {elapsed_since_start:.2f}s since start (no threshold set)."
                )

            run_validation = (
                validation_loader is not None
                and getattr(opt, "validate_every", 0) > 0
                and (epoch % max(1, int(getattr(opt, "validate_every", 1))) == 0)
            )
            val_stats = None
            if run_validation:
                if local_rank == 0:
                    val_stats = _run_validation(
                        net,
                        local_rank,
                        epoch,
                        validation_loader,
                        writer=writer,
                        accuracy_tolerance_px=opt.accuracy_px_tolerance,
                        max_batches=getattr(opt, "val_max_batches", 0),
                        log_path=val_log_path,
                    )
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                if local_rank == 0 and val_stats:
                    val_acc = val_stats.get("accuracy")
                    if val_acc is not None and val_acc > best_val_accuracy:
                        best_val_accuracy = val_acc
                        best_val_epoch = epoch
                        torch.save(net.state_dict(), best_val_path)
                        print(
                            f"[validation] New best accuracy {val_acc * 100:.2f}% "
                            f"at epoch {epoch}. Saved '{best_val_path}'."
                        )

            if runtime_limit_seconds > 0 and not ready_to_exit:
                if elapsed_since_start >= runtime_limit_seconds:
                    ready_to_exit = True
                    opt.ready_to_exit = True
                    if local_rank == 0:
                        print(
                            "[runtime limit] Threshold exceeded; finishing after this epoch. "
                            f"Elapsed {elapsed_since_start:.2f}s (limit {runtime_limit_seconds:.2f}s)."
                        )

            if getattr(opt, "auto_batchsize", False) and auto_batch_ctx is not None:
                opt.batchsize = auto_batch_ctx["controller"].get()
            if auto_batch_ctx and auto_batch_ctx.get("request_restart"):
                raise _AutoBatchRestart(auto_batch_ctx.get("restart_batch_size", opt.batchsize))

            try:
                if local_rank == 0 and epoch % opt.save_every == 0:
                    out_fn = f"{opt.outf}/net_{opt.namefile}_{str(epoch).zfill(4)}.pth"
                    torch.save(net.state_dict(), out_fn)

                    # Clean up old checkpoints if we're limiting the number saved
                    if ckpt_q is not None:
                        if ckpt_q.full():
                            to_del = ckpt_q.get()
                            os.remove(to_del)
                        ckpt_q.put(out_fn)

            except Exception as e:
                print(f"Encountered Exception: {e}")

            if ready_to_exit:
                if local_rank == 0:
                    print(f"[runtime limit] Graceful stop requested; exiting after epoch {epoch}.")
                break

        if local_rank == 0:
            torch.save(
                net.state_dict(),
                f"{opt.outf}/final_net_{opt.namefile}_{str(epoch).zfill(4)}.pth"
            )

        print("end:", datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        print("Total time taken: ", str(datetime.datetime.now() - start_time).split(".")[0])
    except _AutoBatchRestart as restart_exc:
        if getattr(opt, "_restart_done", False):
            print("[auto-batch] Restart already performed once; continuing without another restart.")
        else:
            print(
                f"[auto-batch] Restarting training with stabilized batch size {restart_exc.batch_size}."
            )
            new_opt = copy.deepcopy(opt)
            new_opt.auto_batchsize = False
            new_opt.batchsize = restart_exc.batch_size
            new_opt._restart_done = True
            return main(new_opt)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return


if __name__ == "__main__":
    conf_parser = argparse.ArgumentParser(
        description=__doc__,  # printed with -h/--help
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Turn off help, so we print all options in response to -h
        add_help=False,
    )
    conf_parser.add_argument(
        "-c", "--config",
        help="Specify config file",
        metavar="FILE"
    )
    # Read the config but do not overwrite the args written
    args, remaining_argv = conf_parser.parse_known_args()


    parser = argparse.ArgumentParser()
    # Specify Training Data
    parser.add_argument(
        "--data",
        nargs="+",
        help="Path to training data"
    )
    parser.add_argument(
        "--val_data",
        nargs="+",
        default=None,
        help="Optional validation dataset paths. Leave unset to disable validation.",
    )
    parser.add_argument(
        "--use_s3",
        action="store_true",
        help="Use s3 buckets for training data"
    )
    parser.add_argument(
        "--train_buckets",
        nargs="+",
        default=[],
        help="s3 buckets containing training data. Can list multiple buckets separated by a space.",
    )
    parser.add_argument(
        "--endpoint",
        "--endpoint_url",
        type=str,
        default=None
    )

    # Specify Training Object
    parser.add_argument(
        "--object",
        nargs="+",
        default=None,
        help='Object to train network for. Must match "class" field in groundtruth .json file.'
        ' For best performance, only put one object of interest.',
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="number of data loading workers"
    )
    parser.add_argument(
        "--val_workers",
        type=int,
        default=None,
        help="number of validation loading workers (defaults to --workers)",
    )
    parser.add_argument(
        "--batchsize", "--batch_size",
        type=int,
        default=32,
        help="input batch size"
    )
    parser.add_argument(
        "--val_batchsize",
        type=int,
        default=None,
        help="validation batch size (defaults to --batchsize)",
    )
    parser.add_argument(
        "--auto_batchsize",
        action="store_true",
        help="Estimate the largest batch size that fits into the current GPU memory and override --batchsize."
    )
    parser.add_argument(
        "--auto_batch_headroom",
        type=float,
        default=0.1,
        help=(
            "Fraction of total GPU memory to reserve when using --auto_batchsize "
            "(default: 0.1 -> keep 10% free)."
        ),
    )
    parser.add_argument(
        "--auto_batch_max",
        type=int,
        default=128,
        help="Maximum batch size that auto batching is allowed to reach (default: 128).",
    )
    parser.add_argument(
        "--auto_batch_cooldown",
        type=int,
        default=2,
        help="Number of batch iterations to wait before re-evaluating auto batch size after a change.",
    )
    parser.add_argument(
        "--auto_batch_samples",
        type=int,
        default=5,
        help="Number of steady-state samples to collect before evaluating auto batch size.",
    )
    parser.add_argument(
        "--auto_batch_debug",
        action="store_true",
        help="Print VRAM samples and decisions made by the auto batch estimator.",
    )
    parser.add_argument(
        "--auto_batch_stable_checks",
        type=int,
        default=20,
        help="Disable auto-batching after this many consecutive checks suggest no batch-size change (0 keeps it enabled).",
    )
    parser.add_argument(
        "--imagesize",
        type=int,
        default=448,
        help="the height / width of the input image to network",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate, default=0.0001"
    )
    parser.add_argument(
        "--net_path",
        default=None, help="path to net (to continue training)"
    )
    parser.add_argument(
        "--namefile",
        default="epoch",
        help="name to put on the file of the save weights"
    )
    parser.add_argument(
        "--manualseed",
        type=int,
        help="manual random number seed"
    )
    parser.add_argument(
        "--epochs",
        "--epoch",
        "-e",
        type=int,
        default=60,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--loginterval",
        type=int,
        default=100
    )
    parser.add_argument(
        "--outf",
        default="output/weights",
        help="folder to output images and model checkpoints",
    )
    parser.add_argument(
        "--nb_checkpoints",
        type=int,
        default=0,
        help="Number of checkpoints (.pth files) to save. Older ones will be "
        "deleted as new ones are saved. A value of 0 means an unlimited "
        "number will be saved"
    )
    parser.add_argument(
        '--save_every',
        type=int, default=1,
        help='How often (in epochs) to save a snapshot'
    )
    parser.add_argument(
        "--sigma",
        default=4,
        help="keypoint creation sigma")
    parser.add_argument(
        "--accuracy-px-tolerance",
        type=float,
        default=DEFAULT_KEYPOINT_ACCURACY_TOLERANCE,
        help=(
            "Maximum pixel distance between predicted and ground-truth keypoints "
            "for counting a correct prediction when reporting accuracy."
        ),
    )
    parser.add_argument("--save", action="store_true", help="save a batch and quit")
    parser.add_argument(
        "--max_runtime_seconds",
        type=float,
        default=0,
        help=(
            "Maximum allowed runtime in seconds before requesting a graceful shutdown "
            "after the current epoch finishes (0 disables the limit)."
        ),
    )
    parser.add_argument(
        "--time_threshold_seconds",
        type=float,
        default=0,
        help=(
            "Optional alias for max runtime that is also exposed via the JSON config. "
            "Once elapsed time exceeds the threshold we finish the active epoch and exit."
        ),
    )
    parser.add_argument(
        "--validate_every",
        type=int,
        default=0,
        help="Run validation every N epochs (0 disables validation).",
    )
    parser.add_argument(
        "--val_max_batches",
        type=int,
        default=0,
        help="Limit validation to the first N batches (0 runs the full validation set).",
    )

    config_path = None
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
        if not isinstance(raw_config, dict):
            raise SystemExit(f"[config] JSON config '{config_path}' must contain an object at the top level.")
        valid_dests = {
            action.dest
            for action in parser._actions
            if hasattr(action, "dest") and action.dest not in (argparse.SUPPRESS, "help")
        }
        config_defaults = {
            key: value for key, value in raw_config.items() if key in valid_dests
        }
        ignored = [key for key in raw_config.keys() if key not in valid_dests]
        if ignored:
            print(f"[config] Ignoring unknown option(s) in '{config_path}': {', '.join(ignored)}")
        if config_defaults:
            parser.set_defaults(**config_defaults)
            print(f"[config] Loaded defaults from '{config_path}'. CLI arguments override these values.")
    else:
        print("[config] No JSON config supplied; using CLI defaults only.")

    opt = parser.parse_args(remaining_argv)
    if opt.object is None or len(opt.object) == 0:
        raise SystemExit("--object must be specified via CLI or config.")

    main(opt)
