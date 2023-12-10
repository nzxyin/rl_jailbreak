# rl_jailbreak v0.0.2
Authors: Xavier Yin and Charlie Ji

Affiliation: UC Berkeley EECS, Data Science, Statistics, and Linguistics, Berkeley AI Research

### Installation
```
conda create -n rl_jailbreak python==3.10
conda activate rl_jailbreak
pip install -e .
```

### Testing
#### SFT
```
python sft.py --generator-model [GENERATOR_MODEL_PATH] --target-model [TARGET_MODEL_PATH]
```
#### PPO
```
python main.py --generator-model sft_results/gpt2-xl/checkpoint-6750 --generator-max-tokens 250 --generator-min-tokens 150 --target-model /data/public_models/vicuna/vicuna-7b-v1.3 --target-max-tokens 250 --target-min-tokens 150
```

#### Experiments 12/9
`GPT2-M-SFT vs Llama2-7b-chat`
```
python main.py -r --generator-model sft_results/gpt2-medium/checkpoint-6675 --generator-max-tokens 250 --generator-min-tokens 150 --target-model /data/public_models/llama_v2_chat/Llama-2-7b-chat-hf --target-max-tokens 250 --target-min-tokens 150
```
`GPT2-L-SFT vs Llama2-7b-chat`
```
python main.py -r --generator-model sft_results/gpt2-large/checkpoint-9300 --generator-max-tokens 250 --generator-min-tokens 150 --target-model /data/public_models/llama_v2_chat/Llama-2-7b-chat-hf --target-max-tokens 250 --target-min-tokens 150
```
`GPT2-XL-SFT vs Llama2-7b-chat`
```
python main.py -r --generator-model sft_results/gpt2-xl/checkpoint-6750 --generator-max-tokens 250 --generator-min-tokens 150 --target-model /data/public_models/llama_v2_chat/Llama-2-7b-chat-hf --target-max-tokens 250 --target-min-tokens 150
```

`GPT2-M-SFT vs Zephyr-7b-beta`
```
python main.py -r --generator-model sft_results/gpt2-medium/checkpoint-6675 --generator-max-tokens 250 --generator-min-tokens 150 --target-model /data/public_models/zephyr/zephyr-7b-beta --target-max-tokens 250 --target-min-tokens 150
```
`GPT2-L-SFT vs Zephyr-7b-beta`
```
python main.py -r --generator-model sft_results/gpt2-large/checkpoint-9300 --generator-max-tokens 250 --generator-min-tokens 150 --target-model /data/public_models/zephyr/zephyr-7b-beta --target-max-tokens 250 --target-min-tokens 150
```
`GPT2-XL-SFT vs Zephyr-7b-beta`
```
python main.py -r --generator-model sft_results/gpt2-xl/checkpoint-6750 --generator-max-tokens 250 --generator-min-tokens 150 --target-model /data/public_models/zephyr/zephyr-7b-beta --target-max-tokens 250 --target-min-tokens 150
```



`Vicuna-7b vs Llama2-7b-chat`
```
python main.py -r --generator-model sft_results/vicuna-7b-v1.3-2023-12-10-03-41-55/checkpoint-1350 --generator-max-tokens 250 --generator-min-tokens 150 --target-model /data/public_models/llama_v2_chat/Llama-2-7b-chat-hf --target-max-tokens 250 --target-min-tokens 150
```