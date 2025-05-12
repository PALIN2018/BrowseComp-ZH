for model in Llama4 Gemini2.0-Flash Gemini2.5-Pro DeepSeek-V3 GPT-4o O4-mini O1 Claude3.5-Sonnet DeepSeek-R1 Claude3.7-think Qwen3-235B-thinking Qwen3-235B-no-thinking Qwen2.5-72B-Instruct Qwen2.5-Max QwQ-32B
do
    echo "now running ${model}"
    python run.py --model $model
    wait
done
