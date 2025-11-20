# NoSense baseline for VSR

## Environment setup

```bash
git clone https://github.com/vishaal27/supersanity.git
cd supersanity

python3 -m venv ./env
source env/bin/activate

pip install -r requirements.txt
```

## Data setup
The test datasets are available here: https://huggingface.co/datasets/nyu-visionx/VSI-SUPER-Recall.

To set up the dataset locally:
1. Download the individual `.zip` and `.parquet` files from the Hugging Face page.
2. Unzip the downloaded archives into a single dataset root directory.

After setup, your dataset folder should have the following structure:

```bash
VSI-SUPER-Recall/
├── 10mins/ # Contains 60 short 10-minute video clips
│ ├── video_001.mp4
│ ├── video_002.mp4
│ └── ...
├── 30mins/ # Contains 60 medium-length 30-minute video clips
│ ├── video_001.mp4
│ ├── video_002.mp4
│ └── ...
├── 60mins/ # Contains 60 1-hour video clips
│ ├── video_001.mp4
│ ├── video_002.mp4
│ └── ...
├── 120mins/ # Contains 60 2-hour video clips
│ ├── video_001.mp4
│ ├── video_002.mp4
│ └── ...
├── 240mins/ # Contains 60 4-hour video clips
│ ├── video_001.mp4
│ ├── video_002.mp4
│ └── ...
├── test_10mins.parquet
├── test_30mins.parquet
├── test_60mins.parquet
├── test_120mins.parquet
├── test_240mins.parquet
```

## Running the method
You can run the script using:
```bash
python run_vsi_super_recall.py --model <model-name> --test_split <test-split-name> --gpu_bs <gpu-bs> --data_root <data-root>
```
where
1. `<model-name>` is the name of the CLIP/SigLIP model to use for extracting frame-level features
2. `<test-split-name>` is the test split to run on, can be one of (`10`, `30`, `60`, `120`, `240`)
3. `<gpu-bs>` is the batch-size for computing frame-level features
4. `<data-root>` is the path where the dataset is located
