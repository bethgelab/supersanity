#!/usr/bin/env python3
"""
Utility to perturb the VSC cache by repeating each clip N times.

The script reads videos from HF_HOME/cambrians_vsc/<split> and writes the
perturbed videos to HF_HOME/cambrians_vsc_repeat{N}/<split>. The file names
and metadata stay identical, so the original annotations remain valid.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["10mins"],
        help="Dataset splits to perturb (default: 10mins only).",
    )
    parser.add_argument(
        "--num_repeats",
        nargs="+",
        type=int,
        default=[1, 2, 5],
        help="Number of times to repeat the video (default: 1, 2 and 5 times, so total video length will be 2x, 3x and 6x)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate perturbed files even if they already exist.",
    )
    parser.add_argument(
        "--ffmpeg",
        default=os.environ.get("FFMPEG_BIN", "ffmpeg"),
        help="Path to ffmpeg binary (defaults to $FFMPEG_BIN or 'ffmpeg').",
    )
    return parser.parse_args()


def ensure_ffmpeg_exists(ffmpeg_bin: str):
    if shutil.which(ffmpeg_bin) is None:
        sys.exit(f"ffmpeg binary '{ffmpeg_bin}' not found. Install ffmpeg or set --ffmpeg.")


def perturb_split(ffmpeg_bin: str, src_dir: Path, dst_dir: Path, overwrite: bool, num_repeats: int):
    videos = sorted(src_dir.glob("*.mp4"))
    if not videos:
        print(f"[WARN] No videos found in {src_dir}, skipping.")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Perturbing split '{src_dir.name}' with {len(videos)} videos.")

    for idx, src in enumerate(videos, start=1):
        dst = dst_dir / src.name
        if dst.exists() and not overwrite:
            continue

        cmd = [
            ffmpeg_bin,
            "-y",
            "-v",
            "error",
            "-stream_loop",
            "{}".format(num_repeats),
            "-i",
            str(src),
            "-c",
            "copy",
            str(dst),
        ]
        print(f"[{src_dir.name}] {idx}/{len(videos)} -> {dst.name}")
        subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    ensure_ffmpeg_exists(args.ffmpeg)

    hf_home = Path(os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface/"))).resolve()
    src_base = hf_home / "cambrians_vsc"
    dst_bases = []
    for n in args.num_repeats:
        dst_bases.append( hf_home / "cambrians_vsc_repeat{}".format(n) )

    if not src_base.exists():
        sys.exit(f"Source cache '{src_base}' not found. Download the VSC cache first.")

    for split in args.splits:
        src_dir = src_base / split
        if not src_dir.exists():
            print(f"[WARN] Split '{split}' missing in {src_base}, skipping.")
            continue

        for n, dst_base in zip(args.num_repeats, dst_bases):
            dst_dir = dst_base / split
            perturb_split(args.ffmpeg, src_dir, dst_dir, args.overwrite, num_repeats=n)
            print("[INFO] Perturbation for split {} for num_repeats {} complete.".format(split, n))


if __name__ == "__main__":
    import shutil

    main()
