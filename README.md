# rl_jailbreak
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
Other available parameters: TODO
#### PPO
PPO template
```
python main.py --generator-model [GENERATOR_MODEL_PATH] --target-model [TARGET_MODEL_PATH] 
```
Other available parameters: TODO
e.g.
```
python main.py --generator-model sft_results/gpt2-xl/checkpoint-6750 --generator-max-tokens 250 --generator-min-tokens 150 --target-model /data/public_models/vicuna/vicuna-7b-v1.3 --target-max-tokens 250 --target-min-tokens 150
```