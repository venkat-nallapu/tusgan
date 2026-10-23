#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
watch_and_download.py — Background Kaggle Kernel Watcher & Auto-Downloader
==========================================================================
Monitors a running Kaggle kernel and automatically downloads all output artifacts
(Table 8, Figure 9, model checkpoints, synthetic schedules) directly into the
local repository upon completion.

Usage:
    python watch_and_download.py
    python watch_and_download.py --kernel venkatnallapu/caveat-250-benchmark
    python watch_and_download.py --poll_interval 30
"""

import os
import sys
import time
import zipfile
import argparse
from pathlib import Path
from datetime import datetime

# Root paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
DATA_DIR = SCRIPT_DIR / "data"

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("❌ Error: 'kaggle' package is not installed. Run: pip install kaggle")
    sys.exit(1)


def get_authenticated_api():
    api = KaggleApi()
    api.authenticate()
    return api


def find_active_or_latest_kernel(api, kernel_arg=None):
    """Finds the targeted kernel, or the currently running/queued kernel, or the most recent kernel."""
    if kernel_arg:
        if "/" not in kernel_arg:
            user = api.config_values.get("username", "venkatnallapu")
            kernel_arg = f"{user}/{kernel_arg}"
        return kernel_arg

    print("🔍 Scanning your Kaggle account for active or recent kernels...")
    kernels = api.kernels_list(mine=True, page_size=10)
    if not kernels:
        print("❌ No kernels found for this Kaggle account.")
        return None

    # Check for actively running or queued kernels
    for k in kernels:
        try:
            status_obj = api.kernels_status(k.ref)
            status = str(status_obj.get("status", "")).lower()
            if status in ["running", "queued"]:
                print(f"🎯 Found active kernel: {k.ref} (status: {status.upper()})")
                return k.ref
        except Exception:
            pass

    # Default to the most recently updated kernel
    latest = kernels[0]
    print(f"ℹ️ No currently running kernel detected. Defaulting to most recent kernel: {latest.ref} ({latest.title})")
    return latest.ref


def extract_downloaded_zips(target_dir):
    """Extracts any downloaded zip files in target_dir and cleans up."""
    for zip_path in list(target_dir.glob("*.zip")):
        print(f"📦 Extracting downloaded archive: {zip_path.name}...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(target_dir)
            print(f"   ✓ Extracted successfully to {target_dir}")
        except Exception as e:
            print(f"   ⚠️ Could not extract {zip_path.name}: {e}")

    # If outputs was nested inside outputs/outputs, flatten it
    nested_outputs = target_dir / "outputs"
    if nested_outputs.exists() and nested_outputs.is_dir():
        for item in nested_outputs.iterdir():
            dest = target_dir / item.name
            if dest.exists() and dest.is_dir():
                import shutil
                shutil.rmtree(dest)
            item.rename(dest)
        try:
            nested_outputs.rmdir()
        except Exception:
            pass


def mirror_synthetic_schedules():
    """Copies downloaded synthetic schedules to data/ for local evaluation."""
    for name in ["synth_vae_unconstrained_schedules.csv", "synth_vae_soft_schedules.csv"]:
        found = list(OUTPUTS_DIR.glob(f"**/{name}"))
        if found:
            src = found[0]
            dst = DATA_DIR / name
            import shutil
            shutil.copy2(src, dst)
            print(f"📋 Mirrored schedule: {src.name} → {dst}")


def monitor_and_download(kernel_ref, poll_interval=45):
    api = get_authenticated_api()
    print("=" * 70)
    print("      CAVEAT BENCHMARK AUTOMATED DOWNLOADER & WATCHER")
    print("=" * 70)
    print(f"Target Kernel  : {kernel_ref}")
    print(f"Poll Interval  : {poll_interval}s")
    print(f"Local Dest Dir : {OUTPUTS_DIR}")
    print(f"Started at     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    consecutive_errors = 0
    while True:
        try:
            status_obj = api.kernels_status(kernel_ref)
            status = str(status_obj.get("status", "")).lower()
            fail_msg = status_obj.get("failureMessage")
            now_str = datetime.now().strftime("%H:%M:%S")

            if status == "running":
                print(f"[{now_str}] ⏳ Kernel is RUNNING... (waiting {poll_interval}s)")
            elif status == "queued":
                print(f"[{now_str}] ⏳ Kernel is QUEUED on Kaggle... (waiting {poll_interval}s)")
            elif status == "complete":
                print("\n" + "=" * 70)
                print(f"[{now_str}] 🎉 KERNEL EXECUTION COMPLETE! DOWNLOADING OUTPUTS...")
                print("=" * 70)

                OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
                api.kernels_output(kernel=kernel_ref, path=str(OUTPUTS_DIR), quiet=False)
                print(f"✓ Raw outputs downloaded to {OUTPUTS_DIR}")

                # Extract archives and organize files
                extract_downloaded_zips(OUTPUTS_DIR)
                mirror_synthetic_schedules()

                print("\n📁 Downloaded Outputs Summary:")
                for f in sorted(OUTPUTS_DIR.iterdir()):
                    size_kb = f.stat().st_size / 1024 if f.is_file() else 0
                    print(f"   • {f.name} ({size_kb:.1f} KB)" if f.is_file() else f"   📁 {f.name}/")

                table8_path = OUTPUTS_DIR / "table8_caveat_evaluation.csv"
                fig9_path = OUTPUTS_DIR / "figure9_diurnal_participation.pdf"

                print("\n" + "=" * 70)
                if table8_path.exists():
                    print("✅ Table 8 verified: outputs/table8_caveat_evaluation.csv")
                if fig9_path.exists():
                    print("✅ Figure 9 verified: outputs/figure9_diurnal_participation.pdf")
                print("🏁 All 250-epoch benchmark results are safely on your local disk!")
                print("=" * 70)
                break

            elif status in ["error", "cancel_acknowledged", "canceled"]:
                print(f"\n❌ Kernel ended with status: {status.upper()}")
                if fail_msg:
                    print(f"Failure message: {fail_msg}")
                break
            else:
                print(f"[{now_str}] Status: {status} (waiting {poll_interval}s)")

            consecutive_errors = 0

        except Exception as e:
            consecutive_errors += 1
            print(f"⚠️ Warning checking status (attempt {consecutive_errors}): {e}")
            if consecutive_errors > 10:
                print("❌ Too many consecutive API errors. Exiting.")
                break

        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description="Watch Kaggle kernel and download outputs on completion")
    parser.add_argument("--kernel", type=str, default=None, help="Kaggle kernel slug (e.g. venkatnallapu/caveat-250-benchmark)")
    parser.add_argument("--poll_interval", type=int, default=45, help="Polling interval in seconds (default: 45)")
    args = parser.parse_args()

    api = get_authenticated_api()
    target_kernel = find_active_or_latest_kernel(api, args.kernel)

    if not target_kernel:
        print("Please start the notebook on Kaggle and re-run with: python watch_and_download.py --kernel <name>")
        sys.exit(1)

    monitor_and_download(target_kernel, poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()
