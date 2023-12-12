from rl_jailbreak.models.model import load_generator
from transformers import AutoTokenizer
import pandas as pd
import torch
from tqdm import tqdm
models = {
    "SFT-gpt2-xl": "sft_results/gpt2-xl/checkpoint-6750",
    "SFT-gpt2-l": "sft_results/gpt2-large/checkpoint-9300",
    "SFT-gpt2-m": "sft_results/gpt2-medium/checkpoint-6675",
    "RL-vs-llama-gpt2-xl": "results/PPO-baseline-2023-12-10-08-09-01-EPOCH-0-ITER-100-BATCHSIZE-20",
    "RL-vs-llama-gpt2-l": "results/PPO-baseline-2023-12-10-08-08-58-EPOCH-0-ITER-25-BATCHSIZE-20",
    "RL-vs-llama-gpt2-m": "results/PPO-baseline-2023-12-10-08-08-44-EPOCH-0-ITER-150-BATCHSIZE-20",
    "RL-vs-zephyr-gpt2-xl": "results/PPO-baseline-2023-12-10-08-09-01-EPOCH-0-ITER-100-BATCHSIZE-20",
    "RL-vs-zephyr-gpt2-l": "results/PPO-baseline-2023-12-10-08-08-58-EPOCH-0-ITER-25-BATCHSIZE-20",
    "RL-vs-zephyr-gpt2-m": "results/PPO-baseline-2023-12-10-08-32-03-EPOCH-0-ITER-125-BATCHSIZE-20",
}

generator_tokenizer = AutoTokenizer.from_pretrained("gpt2")

generator_kwargs = {
        "min_length": -1, # don't ignore the EOS token (see above)
        "top_k": 0.0, # no top-k sampling
        "top_p": 1.0, # no nucleus sampling
        "do_sample": True, # yes, we want to sample
        "pad_token_id": generator_tokenizer.eos_token_id, # most decoder models don't have a padding token - use EOS token instead
        "max_new_tokens": 250, # specify how many tokens you want to generate at most
        "min_new_tokens": 150,
    }

data = pd.DataFrame(columns=["model_name", "sample_idx", "text"])

for model_name, model_path in models.items():
    print("Generating prompts for", model_name, "from", model_path)
    generator = load_generator(model_path)
    batch_data = {
            "model_name": [],
            "sample_idx": [],
            "text": []
        }
    for i in tqdm(range(10)):
        generator_output = generator.generate("You are", generator_kwargs)
        batch_data["model_name"].append(model_name)
        batch_data["sample_idx"].append(i)
        batch_data["text"].append(generator_output)

    data = pd.concat([data, pd.DataFrame(batch_data)], ignore_index=True)
    print(data)

data.to_csv("testing/generator_prompts.csv")


