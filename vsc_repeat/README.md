# VSC-Repeat

This part of the repository is essentially a fork of the original Cambrian-S repository from here: https://github.com/cambrian-mllm/cambrian-s. The original README.md file (with some minimal modifications) is here: https://github.com/bethgelab/supersanity/blob/main/vsc_repeat/README_original.md. We describe the main set of steps required to reproduce the VSC-Repeat results here and defer other details to the original README and repository.

## Installation

```bash
# Create conda environment
conda create --name cambrians_eval python=3.10
conda activate cambrians_eval

# Clone the repository
git clone git@github.com:bethgelab/supersanity.git
cd supersanity/vsc_repeat/lmms-eval

# Install lmms-eval
pip install -e .

# Install flash-attn for faster inference (recommended)
pip install flash-attn==2.8.3 --no-build-isolation
```

## Running VSC-Repeat experiments
We provide a script to generate a perturbation where each clip in the 10-minute VSC split is repeated N times (N=1, 2 or 5, yielding 20, 30 or 60 minutes total) while reusing the original ground-truth annotations. This is useful for testing sensitivity to longer inputs without modifying the benchmark metadata.

1. **Generate the repeated cache**
   ```bash
   cd supersanity/vsc_repeat/lmms-eval
   export HF_HOME=/path/to/your/hf_cache           # same cache used for VSI-SUPER
   python scripts/perturb_vsc_repeat.py --splits 10mins
   ```
   This creates `HF_HOME/cambrians_vsc_repeat{N}/10mins` by looping each `.mp4` N times via `ffmpeg -stream_loop {N}`, where N=1, 2 or 5.

2. **Run the VSC-repeat benchmark**
   ```bash
   export HF_HOME=/path/to/your/hf_cache
   export CAMBRIANS_VSC_CACHE_NAME=cambrians_vsc_repeat{N}
   bash scripts/vsc_repeat{N}.sh
   ```
   `scripts/vsc_repeat{N}.sh` launches `evaluate_all_in_one.sh` on the new task `cambrians_vsc_repeat{N}_10mins`. It is critical to set the correct `CAMBRIANS_VSC_CACHE_NAME` path pointing to the correct set of repeated videos, since this environment variable is used for setting the path from which the videos are read and loaded. Results are logged under `logs/<checkpoint>/cambrians_vsc_repeat{N}_10mins`, making it easy to compare with the original `cambrians_vsc_10mins`. Replace {N} with 1, 2 or 5 for the desired number of repeats in the repeated experiment.


