#!/usr/bin/env bash

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
else
    export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
fi
export DECORD_EOF_RETRY_MAX=20480 # videommu and hourvideo require this

# Default to repeat3 cache if user hasn't set one explicitly.
export CAMBRIANS_VSC_CACHE_NAME=${CAMBRIANS_VSC_CACHE_NAME:-cambrians_vsc_repeat3}

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    num_processes=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
else
    IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
    num_processes=${#devices[@]}
fi

checkpoint=ShushengYang/Cambrian-S-7B-LFP
echo "Evaluating checkpoint: $checkpoint on benchmark cambrians_vsc_repeat3_10mins"
echo "Using cache: ${CAMBRIANS_VSC_CACHE_NAME}"

bash evaluate_all_in_one.sh --model cambrians_vsc --benchmark cambrians_vsc_repeat3_10mins --num_processes ${num_processes:-1} --num_frames -1 --pretrained $checkpoint --miv_token_len ${MIV_TOKEN_LEN:-64} --si_token_len ${SI_TOKEN_LEN:-729} --sensory_window_size 128 --enable_visual_feature_caching True --surprise_threshold 0.39
