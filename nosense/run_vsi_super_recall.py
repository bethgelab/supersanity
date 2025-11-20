import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader, cpu
from tqdm import tqdm
from open_clip import create_model_from_pretrained, get_tokenizer
from torch.amp import autocast
import torch
import numpy as np
from argparse import ArgumentParser

# where to load the open-clip weights from
try:
    CACHE_DIR = os.environ["HF_HOME"]
except KeyError as e:
    try:
        CACHE_DIR = os.environ["TORCH_HOME"]
    except KeyError as e:
        CACHE_DIR = "~/.cache"

CLIP_MODELS = {
    "clip_vitl14_224": "hf-hub:timm/vit_large_patch14_clip_224.openai",
    "siglip2_b16_224": "hf-hub:timm/ViT-B-16-SigLIP2",
    "siglip2_so400m16_384": "hf-hub:timm/ViT-SO400M-16-SigLIP2-384",
    "siglip2_so400m16_512": "hf-hub:timm/ViT-SO400M-16-SigLIP2-512",
    "siglip2_b32_256": "hf-hub:timm/ViT-B-32-SigLIP2-256",
    "dfn5b_h14_384": "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384",
    "datology_cls": "hf-hub:DatologyAI/cls-opt-vit-b-32",
    "datology_ret": "hf-hub:DatologyAI/retr-opt-vit-b-32",
}
PATHS_TO_NAME_MAPPING = {v:k for k,v in CLIP_MODELS.items()}

# we hardcode this to 1FPS
FPS_TARGET = 1

ANSWER_MAPPING = {
    "A" : 0,
    "B": 1,
    "C": 2,
    "D": 3,
}

# create multiple random prompts describing the same object
prompt_templates = [
    "a photo of {}",
    "a picture of {}",
    "an image showing {}",
    "a clear view of {}",
    "a snapshot of {}",
    "a video frame depicting {}",
    "a close-up of {}",
    "a realistic photo of {}",
    "a detailed view of {}",
    "{} is in the scene",
    "the frame contains {}",
    "{}",
    "{} ",
    "{} is in the photo",
    "what is in the photo? {}",
    "wow what a cute photo of {}",
]

# create multiple random prompts describing the main object and side object of interest
relation_templates = [
    "{} is on top of {}",
    "{} is near {}",
    "{} is next to {}",
    "{} is below {}",
    "{} is in front of {}",
    "{} is behind {}",
    "{} is beside {}",
    "{} is close to {}",
    "{} is far from {}",
    "{} is leaning against {}",
    "{} is touching {}",
    "{} is overlapping with {}",
    "{} is surrounded by {}",
    "{} is facing {}",
    "{} is approaching {}",
    "{} is interacting with {}",
    "{} is above {}",
    "{} is under {}",
    "{} appears with {} in the frame",
    "{} and {} are seen together",
]

# === DATASET ===
class VideoFrameDataset(Dataset):
    def __init__(self, parquet_path, videos_root, fps_target=1):
        self.df = pd.read_parquet(parquet_path)
        self.videos_root = videos_root
        self.fps_target = fps_target

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = os.path.join(self.videos_root, row["video_path"])

        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            native_fps = vr.get_avg_fps()
            step = int(round(native_fps / self.fps_target))

            # grab 1 FPS frames
            frame_buffers = []
            for i in range(0, len(vr), step):
                frame = vr[i].asnumpy()
                image = Image.fromarray(frame)
                frame_buffers.append(image)

            meta = {
                "video_path": str(row["video_path"]),
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "options": list(row["options"]),
                "num_frames": len(frame_buffers),
            }

            return {"frames": frame_buffers, "meta": meta}

        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            return None

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=list(CLIP_MODELS.keys()))
    parser.add_argument('--test_split', type=int, required=True, choices=[10, 30, 60, 120, 240])
    parser.add_argument('--gpu_bs', type=int, required=False, default=512)
    parser.add_argument('--data_root', type=str, required=False, default="/p/data1/datasets/mmlaion/vudandar_tests/vsi_benchmark/vsi-super-recall/")
    args = parser.parse_args()

    parquet_path = f"{args.data_root}test_{args.test_split}mins.parquet"

    # load clip model
    model, preprocess = create_model_from_pretrained(CLIP_MODELS[args.model], cache_dir=CACHE_DIR, device="cuda")
    tokenizer = get_tokenizer(CLIP_MODELS[args.model], cache_dir=CACHE_DIR)
    if torch.__version__ >= "2.0":
        try:
            model = torch.compile(model)
        except Exception as e:
            raise e
    model.eval()

    # get video dataloader
    dataset = VideoFrameDataset(parquet_path, args.data_root, FPS_TARGET)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=1,
        collate_fn=lambda x: x[0],
    )

    # for tracking accuracy
    correct = 0
    total = 0

    with torch.inference_mode(), autocast("cuda"):

        # iterate through full test set
        for batch_index, sample in tqdm(enumerate(dataloader), ascii=True, total=len(dataloader), desc="Running..."):

            # assume bs=1
            assert isinstance(sample, dict)
            assert "meta" in sample
            assert "frames" in sample

            # get test sample
            m = sample["meta"]
            question = m["question"]
            answer = m["answer"]
            opts = m["options"]
            frames = sample["frames"]

            # first, get the object of interest
            obj = question.replace("These are frames of a video.\nWhich of the following correctly represents the order in which", "")
            obj = obj.replace("appeared in the video?", "")
            obj = obj.replace("the ", "").strip()

            # get object embedding (ensembled)
            prompts = [p.format(obj.strip()) for p in prompt_templates]
            texts_ = tokenizer(prompts, context_length=model.context_length).to("cuda")
            text_features_all = model.encode_text(texts_, normalize=True)
            text_features = text_features_all.mean(dim=0, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # get all answer choice objects
            objects = [x.strip() for x in opts[0].replace("A. ", "").split(",")]

            # compute 4 relational embeddings (one per option)
            relational_embeddings = []
            for other_obj in objects:
                prompts_rel = [t.format(obj.strip(), other_obj.strip()) for t in relation_templates]
                texts_rel = tokenizer(prompts_rel, context_length=model.context_length).to("cuda")
                text_feats_rel_all = model.encode_text(texts_rel, normalize=True)
                text_feats_rel = text_feats_rel_all.mean(dim=0, keepdim=True)
                text_feats_rel = text_feats_rel / text_feats_rel.norm(dim=-1, keepdim=True)
                relational_embeddings.append(text_feats_rel)

            # get full concatenated text embedding (main object + 4 main obj w/ option obj)
            text_features_final = torch.cat([text_features] + relational_embeddings, dim=0)

            # get frame embeddings
            frame_features = []
            for i in range(0, len(frames), args.gpu_bs):
                fs = frames[i : i + args.gpu_bs]
                images_ = torch.stack([preprocess(image) for image in fs]).to("cuda")
                feats = model.encode_image(images_, normalize=True)
                frame_features.append(feats)
            frame_features = torch.cat(frame_features, dim=0)

            # get sims
            sims = (frame_features @ text_features_final.t())
            obj_only_sims = sims[:, 0]

            # get the top 4 cosine sims with the main object
            top_values, top_indices = torch.topk(obj_only_sims, k=4, dim=0)

            # get the top indices, sort them in ascending order and get the corresponding main object-sub-object scores
            top_indices_sorted = torch.sort(top_indices).values
            relation_scores = sims[top_indices_sorted, 1:]

            # get individual answer positions
            ind_opt_positions = []
            for opt in opts:
                ind_objs = [x.strip() for x in opt.replace("A. ", "").replace("B. ", "").replace("C. ", "").replace("D. ", "").split(",")]
                ind_objs_pos = [objects.index(x) for x in ind_objs]
                ind_opt_positions.append(ind_objs_pos)

            # get per-option score sums
            opt_scores = []
            for pos in ind_opt_positions:
                pos_tensor = torch.tensor(pos, device=relation_scores.device)
                # gather one score per frame according to option's ordering
                ordered_scores = relation_scores[torch.arange(4), pos_tensor]
                opt_scores.append(ordered_scores.sum().item())

            # update accuracy
            total += 1
            if np.argmax(opt_scores) == ANSWER_MAPPING[answer]:
                correct += 1

    print("Current model: {}".format(args.model))
    print("Current test split: {}".format(args.test_split))
    print("Accuracy: {}% (correct={}/total={})".format(100. * correct/total, correct, total))
