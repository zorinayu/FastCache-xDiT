#!/bin/bash
# FastCache direct test script - bypasses registration mechanism

# Default model type is SD3
MODEL_TYPE="sd3"
MODEL="stabilityai/stable-diffusion-3-medium-diffusers"
PROMPT="a serene landscape with mountains and a lake"
NUM_STEPS=30
CACHE_RATIO=0.05
MOTION_THRESHOLD=0.1

# Process command line arguments
if [ "$1" == "flux" ]; then
    MODEL_TYPE="flux"
    MODEL="black-forest-labs/FLUX.1-schnell"
fi

# Check if number of steps is specified
if [ ! -z "$2" ]; then
    NUM_STEPS=$2
fi

# Add project root directory to Python path
export PYTHONPATH=$PWD:$PYTHONPATH

echo "==================== FastCache Test ===================="
echo "Model type: $MODEL_TYPE"
echo "Model: $MODEL"
echo "Prompt: $PROMPT"
echo "Steps: $NUM_STEPS"
echo "Cache ratio threshold: $CACHE_RATIO"
echo "Motion threshold: $MOTION_THRESHOLD"
echo "======================================================="

# Run the test script
python examples/run_fastcache_test.py \
    --model_type $MODEL_TYPE \
    --model $MODEL \
    --prompt "$PROMPT" \
    --num_inference_steps $NUM_STEPS \
    --cache_ratio_threshold $CACHE_RATIO \
    --motion_threshold $MOTION_THRESHOLD

echo "======================================================="
echo "Test complete!" 