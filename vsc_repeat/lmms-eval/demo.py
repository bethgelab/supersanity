import sys

sys.path = ["../"] + sys.path

from argparse import ArgumentParser
from PIL import Image
import torch
import numpy as np

from cambrian.constants import IMAGE_TOKEN_INDEX
from cambrian.conversation import conv_templates
from cambrian.model.builder import load_pretrained_model
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, expand2square

from decord import VideoReader, cpu


def process_video_with_decord(video_file, model_cfg, num_threads=-1):

    if num_threads < 1:
        vr = VideoReader(video_file, ctx=cpu(0))
    else:
        vr = VideoReader(video_file, ctx=cpu(0), num_threads=num_threads)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    avg_fps = round(vr.get_avg_fps() / model_cfg.video_fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]
    frame_time = [i / avg_fps for i in frame_idx]

    if model_cfg.video_max_frames > 0:
        if len(frame_idx) > model_cfg.video_max_frames or model_cfg.video_force_sample:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, model_cfg.video_max_frames, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / vr.get_avg_fps() for i in frame_idx]

    video = vr.get_batch(frame_idx).asnumpy()
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

    num_frames_to_sample = num_frames = len(frame_idx)
    vr.seek(0)
    return video, video_time, frame_time, num_frames_to_sample


def process_videos(videos, image_processor, model_cfg, num_threads=-1):

    processor_aux_list = image_processor

    new_videos_aux_list = []
    video_sizes = []

    for video in videos:
        video, video_time, frame_time, num_frames_to_sample = process_video_with_decord(video, model_cfg, num_threads=num_threads)
        video_sizes.append((video.shape[2], video.shape[1], video.shape[0]))  # W, H, T
        video = [Image.fromarray(video[_], mode="RGB") for _ in range(video.shape[0])]  # covert to PIL.Image.Image

        video_aux_list = []
        for processor_aux in processor_aux_list:
            video_aux = video
            video_aux = [expand2square(image, tuple(int(x * 255) for x in processor_aux.image_mean)) for image in video_aux]
            video_aux_list.append(processor_aux.preprocess(video_aux, return_tensors="pt")["pixel_values"])

        new_videos_aux_list.append(video_aux_list)

    new_videos_aux_list = [list(batch_video_aux) for batch_video_aux in zip(*new_videos_aux_list)]
    new_videos_aux_list = [torch.stack(video_aux) for video_aux in new_videos_aux_list]

    return new_videos_aux_list, video_sizes, (video_time, frame_time, num_frames_to_sample)


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_image", type=str, default=None)
    parser.add_argument("--input_video", type=str, default=None)
    parser.add_argument("--model", type=str, default="nyu-visionx/Cambrian-S-0.5B")
    parser.add_argument("--question", type=str, required=True)

    parser.add_argument("--video_max_frames", type=int, default=128)
    parser.add_argument("--video_fps", type=int, default=1)
    parser.add_argument("--video_force_sample", action="store_true", default=False)
    parser.add_argument("--miv_token_len", type=int, default=64)
    parser.add_argument("--si_token_len", type=int, default=729)
    parser.add_argument("--image_aspect_ratio", type=str, default="anyres")
    parser.add_argument("--anyres_max_subimages", type=int, default=9)

    parser.add_argument("--conv_template", type=str, default="qwen_2")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)

    args = parser.parse_args()

    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model, None, get_model_name_from_path(args.model), device_map="cuda:0")

    model.config.video_max_frames = args.video_max_frames
    model.config.video_fps = args.video_fps
    model.config.video_force_sample = args.video_force_sample
    model.config.miv_token_len = args.miv_token_len
    model.config.si_token_len = args.si_token_len
    model.config.image_aspect_ratio = args.image_aspect_ratio
    model.config.anyres_max_subimages = args.anyres_max_subimages

    assert args.input_image or args.input_video, "Either input_image or input_video must be provided"
    assert not (args.input_image and args.input_video), "Only one of input_image or input_video can be provided"

    if args.input_image is not None:
        visual_tensors, visual_sizes = process_images([Image.open(args.input_image).convert("RGB")], image_processor, model.config)
    elif args.input_video is not None:
        visual_tensors, visual_sizes, (_, _, _) = process_videos([args.input_video], image_processor, model.config)
    else:
        raise ValueError("Invalid input type")

    assert "<image>" not in args.question
    question = "<image>\n" + args.question
    conv = conv_templates[args.conv_template].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_beams": args.num_beams,
    }

    with torch.inference_mode():
        input_ids = input_ids.cuda()

        visual_tensors = [_.half().cuda() for _ in visual_tensors]

        output_ids = model.generate(
            inputs=input_ids,
            images=visual_tensors,
            image_sizes=visual_sizes,
            use_cache=True,
            do_sample=True if gen_kwargs["temperature"] > 0 else False,
            temperature=gen_kwargs["temperature"],
            top_p=gen_kwargs["top_p"],
            num_beams=gen_kwargs["num_beams"],
            max_new_tokens=gen_kwargs["max_new_tokens"],
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)


if __name__ == "__main__":
    main()
