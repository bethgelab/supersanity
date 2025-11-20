#!/bin/bash

set -e

export DECORD_EOF_RETRY_MAX=20480

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
else
    IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
    gpu_count=${#devices[@]}
fi

benchmark=""
num_processes=4
num_frames=64
launcher=accelerate
pretrained=""
main_process_port=12345
miv_token_len=64
si_token_len=729

available_models="cambrians"

while [[ $# -gt 0 ]]; do
    case "$1" in
    --benchmark)
        benchmark="$2"
        shift 2
        ;;
    --num_processes)
        num_processes="$2"
        shift 2
        ;;
    --model)
        IFS=',' read -r -a models <<<"$2"
        shift 2
        ;;
    --output_path)
        output_path="$2"
        shift 2
        ;;
    --limit)
        limit="$2"
        shift 2
        ;;
    --num_frames)
        num_frames="$2"
        shift 2
        ;;
    --pretrained)
        pretrained="$2"
        shift 2
        ;;
    --main_process_port)
        main_process_port="$2"
        shift 2
        ;;
    --miv_token_len)
        miv_token_len="$2"
        shift 2
        ;;
    --si_token_len)
        si_token_len="$2"
        shift 2
        ;;
    --sensory_window_size)
        sensory_window_size="$2"
        shift 2
        ;;
    --compression_downsample_ratio)
        compression_downsample_ratio="$2"
        shift 2
        ;;
    --surprise_threshold)
        surprise_threshold="$2"
        shift 2
        ;;
    --consolidation_method)
        consolidation_method="$2"
        shift 2
        ;;
    --retrieval_topk)
        retrieval_topk="$2"
        shift 2
        ;;
    --enable_visual_feature_caching)
        enable_visual_feature_caching="$2"
        shift 2
        ;;
    --consolidation_mem_budget)
        consolidation_mem_budget="$2"
        shift 2
        ;;
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
    esac
done

if [ "$models" = "all" ]; then
    IFS=',' read -r -a models <<<"$available_models"
fi

for model in "${models[@]}"; do
    echo "Start evaluating $model..."

    case "$model" in
    "cambrians")
        model_family="cambrians"
        model="cambrians_${pretrained}_${num_frames}f"
        model_args="pretrained=${pretrained},conv_template=qwen_2,video_max_frames=${num_frames},miv_token_len=${miv_token_len},si_token_len=${si_token_len}"
        ;;
    "cambrians_vsr")
        model_family="cambrians_vsr"
        model="cambrians_vsr_${pretrained}"
        model_args="pretrained=${pretrained},conv_template=qwen_2,miv_token_len=${miv_token_len},si_token_len=${si_token_len},sensory_window_size=${sensory_window_size},compression_downsample_ratio=${compression_downsample_ratio},consolidation_method=${consolidation_method},retrieval_topk=${retrieval_topk},enable_visual_feature_caching=${enable_visual_feature_caching},surprise_threshold=${surprise_threshold},consolidation_mem_budget=${consolidation_mem_budget}"
        ;;
    "cambrians_vsc")
        model_family="cambrians_vsc"
        model="cambrians_vsc_${pretrained}"
        model_args="pretrained=${pretrained},conv_template=qwen_2,miv_token_len=${miv_token_len},si_token_len=${si_token_len},sensory_window_size=${sensory_window_size},enable_visual_feature_caching=${enable_visual_feature_caching},surprise_threshold=${surprise_threshold}"
        ;;
    "cambrians_vsc_streaming")
        model_family="cambrians_vsc_streaming"
        model="cambrians_vsc_streaming_${pretrained}"
        model_args="pretrained=${pretrained},conv_template=qwen_2,miv_token_len=${miv_token_len},si_token_len=${si_token_len},sensory_window_size=${sensory_window_size},enable_visual_feature_caching=${enable_visual_feature_caching},surprise_threshold=${surprise_threshold}"
        ;;
    *)
        echo "Unknown model: $model"
        exit -1
        ;;
    esac

    if [ "$launcher" = "python" ]; then
        export LMMS_EVAL_LAUNCHER="python"
        evaluate_script="python \
            "
    elif [ "$launcher" = "accelerate" ]; then
        export LMMS_EVAL_LAUNCHER="accelerate"
        evaluate_script="accelerate launch \
            --num_processes=$num_processes \
            --main_process_port $main_process_port \
            "
    fi

    if [ -z "$output_path" ]; then
        output_path="logs/$(basename $pretrained)/$benchmark"
    fi

    evaluate_script="$evaluate_script -m lmms_eval \
        --model $model_family \
        --model_args $model_args \
        --tasks $benchmark \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix $model \
        --output_path $output_path \
        "

    if [ -n "$limit" ]; then
        evaluate_script="$evaluate_script \
            --limit $limit \
        "
    fi
    echo $evaluate_script
    eval $evaluate_script
done
