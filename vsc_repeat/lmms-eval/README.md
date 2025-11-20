# Cambrian-S Evaluation

## Overview

This directory provides a comprehensive toolkit for evaluating Cambrian-S models. Our evaluation suite is adapted from the [LMMS-EVAL](https://github.com/EvolvingLMMs-Lab/lmms-eval) framework.

Currently supported benchmarks include:
- VSI-Bench
- VSI-SUPER (Count and Recall)
- MMVP
- EgoSchema
- MVBench
- VideoMME
- VideoMMMU
- LongVideoBench

For a detailed list of tasks, please refer to the [`lmms_eval/tasks/`](lmms_eval/tasks/) directory.

## Setup

### Prerequisites

Follow the instructions below to set up your Python environment:

```bash
# Create conda environment
conda create --name cambrians_eval python=3.10
conda activate cambrians_eval

# Clone the repository
git clone git@github.com:cambrian-mllm/cambrian-s.git
cd cambrian-s

# Install Cambrian-S
pip install -e .

# Install lmms-eval
cd lmms-eval
pip install -e .

# Install flash-attn for faster inference (recommended)
pip install flash-attn==2.8.3 --no-build-isolation
```

## Evaluation

### Cambrian-S

Evaluate Cambrian-S models w/o predictive sensing:

```bash
cd cambrian-s/lmms-eval

# Set GPU devices (optional, defaults to 0-7)
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi

# Required for videommu and hourvideo benchmarks
export DECORD_EOF_RETRY_MAX=20480

# Automatically detect number of GPUs
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    num_processes=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
else
    IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
    num_processes=${#devices[@]}
fi

# Configuration
checkpoint="nyu-visionx/Cambrian-S-7B" 
benchmark="vsibench" 

# Run evaluation
bash evaluate_all_in_one.sh \
    --model cambrians \
    --benchmark "$benchmark" \
    --num_processes ${num_processes:-1} \
    --num_frames ${NUM_FRAMES:-128} \
    --pretrained $checkpoint \
    --miv_token_len ${MIV_TOKEN_LEN:-64} \
    --si_token_len ${SI_TOKEN_LEN:-729}
```

#### Parameters

- `NUM_FRAMES`: Maximum number of frames to sample uniformly from the input video (default: 128)
- `MIV_TOKEN_LEN`: Number of tokens per video frame (default: 64)
- `SI_TOKEN_LEN`: Number of tokens per independent input image (default: 729)

For complete benchmark results, please refer to our paper.

### Cambrian-S (with Predictive Sensing) & VSI-SUPER

Evaluate Cambrian-S-7B-LFP with predictive sensing on VSI-SUPER benchmarks.

```bash
# VSI-SUPER Recall
bash eval_vsr.sh

# VSI-SUPER Count
bash eval_vsc.sh

# VSI-SUPER Count (Streaming)
bash eval_vsc_streaming.sh 
```

### VSC Repeat-N Perturbation

We provide an optional perturbation where each clip in the 10-minute VSC split is repeated N times (N=1, 2 or 5, yielding 20, 30 or 60 minutes total) while reusing the original ground-truth annotations. This is useful for testing sensitivity to longer inputs without modifying the benchmark metadata.

1. **Generate the perturbed cache**
   ```bash
   cd cambrian-s/lmms-eval
   export HF_HOME=/path/to/your/hf_cache           # same cache used for VSI-SUPER
   python3 scripts/perturb_vsc_repeat.py --splits 10mins
   ```
   This creates `HF_HOME/cambrians_vsc_repeat{N}/10mins` by looping each `.mp4` N times via `ffmpeg -stream_loop {N}`, where N=1, 2 or 5.

2. **Run the repeat-N benchmark**
   ```bash
   export HF_HOME=/path/to/your/hf_cache
   export CAMBRIANS_VSC_CACHE_NAME=cambrians_vsc_repeat{N}
   bash scripts/vsc_repeat{N}.sh
   ```
   `scripts/vsc_repeat{N}.sh` launches `evaluate_all_in_one.sh` on the new task `cambrians_vsc_repeat{N}_10mins`. Results are logged under `logs/<checkpoint>/cambrians_vsc_repeat{N}_10mins`, making it easy to compare with the original `cambrians_vsc_10mins`. Replace {N} with 1, 2 or 5 for the desired number of repeats in the perturbed test.

## Demo

We also provide one demo script with which you can quickly try our model on both image and video, hope it helps!

```bash

# Image QA
python demo.py --question "Please describe this image in detail" --input_image ../figs/feature.png --max_new_tokens 512

# Video QA
python demo.py --question "Please describe this video in detail" --input_video $VIDEO_PATH --max_new_tokens 512
```

## Acknowledgement

We sincerely thank [LMMS-EVAL](https://github.com/EvolvingLMMs-Lab/lmms-eval) for providing this convenient evaluation toolkit.
