from rl_jailbreak.models.model import load_generator, load_target, load_reward
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, TrainingArguments
from tqdm import tqdm      
import argparse  
import pathlib
import pandas as pd
from datasets import Dataset
import os 
from peft import LoraConfig


def main(args):

    # Handle Logging
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    df = pd.read_csv(args.dataset)    
    # rename the column "prompt" to "text"
    df = df.rename(columns={"prompt": "text"})
    
    # remove the other columns 
    df = df.drop(columns=["platform", "source", "jailbreak", "created_at", "date", "community_id", "community_name"])

    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(args.sft_eval_size)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLM.from_pretrained(args.generator_model)

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        logging_strategy='epoch',
        save_strategy='epoch',
        evaluation_strategy='epoch',
        save_total_limit=1,
        warmup_ratio=0.1,
        per_device_train_batch_size=args.sft_batch_size,
        per_device_eval_batch_size=args.sft_batch_size,
        load_best_model_at_end=True,
        num_train_epochs=args.sft_num_epochs,
    )

    trainer = SFTTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field="text", # TODO: header name here
        peft_config=peft_config, 
        max_seq_length=args.sft_max_seq_len,
    )

    trainer.train()

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    ########### Generator model parameters ##########
    parser.add_argument(
        "--generator-model", 
        default = "gpt2-medium",
        help = "Name of attack generator model.",
        choices=["gpt2-medium"]
    )
    
    ##################################################

    ########### Dataset parameters ##########
    parser.add_argument(
        "--dataset",
        default="./datasets/sft_dataset.csv",
        help = "Path to SFT dataset.",
        type = pathlib.Path,
    )
    
    ##################################################

    ########### LoRA parameters ##########
    parser.add_argument(
        "--lora-r",
        default = 16,
        help = "Rank of update matrix for LoRA",
        type = int,
    )

    parser.add_argument(
        "--lora-alpha",
        default = 32,
        help = "Alpha value for LoRA",
        type = float,
    )

    parser.add_argument(
        "--lora-dropout",
        default = 0.05,
        help = "Dropout for LoRA",
        type = float,
    )
    
    # TODO: add other PPO params
    ##################################################
    
    ########### Logging parameters ##########
    
    parser.add_argument(
        "--save_dir",
        default="./sft_results",
        help = "Path to save the model.",
        type = pathlib.Path,
    )

    ##################################################

    ########### SFT parameters ##########
    
    parser.add_argument(
        "--sft-max-seq-len",
        default=256,
        help = "Maximum sequence length for SFT",
        type = int,
    )

    parser.add_argument(
        "--sft-num_epochs",
        default=100.,
        help = "Num epochs for SFT",
        type = float,
    )

    parser.add_argument(
        "--sft-lr",
        default=5e-4,
        help = "Learning Rate for SFT",
        type = float,
    )

    parser.add_argument(
        "--sft-batch-size",
        default=8,
        help = "Batch Size for SFT",
        type = int,
    )

    parser.add_argument(
        "--sft-eval-size",
        default=0.1,
        help = "Eval Dataset Size for SFT",
        type = float,
    )

    ##################################################

    args = parser.parse_args()
    main(args)
    
