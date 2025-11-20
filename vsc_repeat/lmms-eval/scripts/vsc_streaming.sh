
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
else
    export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
fi
export DECORD_EOF_RETRY_MAX=20480 # videommu and hourvideo require this

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    num_processes=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
else
    IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
    num_processes=${#devices[@]}
fi

checkpoint="ShushengYang/Cambrian-S-7B-LFP"

bash evaluate_all_in_one.sh --model cambrians_vsc_streaming --benchmark cambrians_vsc_streaming_10mins --num_processes ${num_processes:-1} --num_frames -1 --pretrained $checkpoint --miv_token_len ${MIV_TOKEN_LEN:-64} --si_token_len ${SI_TOKEN_LEN:-729} --sensory_window_size 128 --enable_visual_feature_caching True --surprise_threshold 0.39

bash evaluate_all_in_one.sh --model cambrians_vsc_streaming --benchmark cambrians_vsc_streaming_30mins --num_processes ${num_processes:-1} --num_frames -1 --pretrained $checkpoint --miv_token_len ${MIV_TOKEN_LEN:-64} --si_token_len ${SI_TOKEN_LEN:-729} --sensory_window_size 128 --enable_visual_feature_caching True --surprise_threshold 0.39

bash evaluate_all_in_one.sh --model cambrians_vsc_streaming --benchmark cambrians_vsc_streaming_60mins --num_processes ${num_processes:-1} --num_frames -1 --pretrained $checkpoint --miv_token_len ${MIV_TOKEN_LEN:-64} --si_token_len ${SI_TOKEN_LEN:-729} --sensory_window_size 128 --enable_visual_feature_caching True --surprise_threshold 0.39

bash evaluate_all_in_one.sh --model cambrians_vsc_streaming --benchmark cambrians_vsc_streaming_120mins --num_processes ${num_processes:-1} --num_frames -1 --pretrained $checkpoint --miv_token_len ${MIV_TOKEN_LEN:-64} --si_token_len ${SI_TOKEN_LEN:-729} --sensory_window_size 128 --enable_visual_feature_caching True --surprise_threshold 0.41

