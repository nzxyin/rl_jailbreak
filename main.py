from rl_jailbreak.models.model import load_generator, load_target, load_reward
from trl import PPOTrainer, PPOConfig
from tqdm import tqdm      
import argparse  
import pathlib
import pandas as pd
from datasets import Dataset
import os 
import torch
from datetime import datetime
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
from torch.utils.tensorboard import SummaryWriter


def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    run_name = f"{args.experiment_name}-{datetime.now().strftime('%Y-%m-%d|%H:%M:%S')}"
    
    ppo_config = PPOConfig(
        model_name=args.generator_model,
        learning_rate=args.ppo_lr,
        batch_size=args.ppo_batch_size,
        log_with="tensorboard",
        
        project_kwargs={
            "logging_dir":f"{args.log_dir}/{run_name}",
        },
        
    )
    writer = SummaryWriter(f"{args.log_dir}/{run_name}")
    # generator = load_generator(args.generator_model)


    generator_tokenizer = AutoTokenizer.from_pretrained(args.generator_model, padding_side='left')
    generator_tokenizer.pad_token = generator_tokenizer.eos_token
    generator_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.generator_model, device_map="auto")

    target = load_target(args.target_model)
    reward_model = load_reward(args.reward_model)

    df = pd.read_csv(args.dataset)    
    # rename the column "question" to "query"
    df = df.rename(columns={"question": "query"})
    
    # remove the other columns risk_area,types_of_harm,source,new_category
    df = df.drop(columns=["Unnamed: 0", "risk_area", "types_of_harm", "source", "new_category"])

    dataset = Dataset.from_pandas(df)
    # TODO encode dataset using target model and slice <BOS> token off

    generator_kwargs = {
        "min_length": -1, # don't ignore the EOS token (see above)
        "top_k": 0.0, # no top-k sampling
        "top_p": 1.0, # no nucleus sampling
        "do_sample": True, # yes, we want to sample
        "pad_token_id": generator_tokenizer.eos_token_id, # most decoder models don't have a padding token - use EOS token instead
        "max_new_tokens": args.generator_max_tokens, # specify how many tokens you want to generate at most
        "min_new_tokens": args.generator_min_tokens,
    }

    target_kwargs = {
        "do_sample": True, # yes, we want to sample
        "pad_token_id": target.tokenizer.eos_token_id, # most decoder models don't have a padding token - use EOS token instead
        "max_new_tokens": args.target_max_tokens, # specify how many tokens you want to generate at most
        "min_new_tokens": args.target_min_tokens,
        "repetition_penalty": 2.0,
    }

    ppo_trainer = PPOTrainer(
        model=generator_model,
        config=ppo_config,
        dataset=dataset,
        tokenizer=generator_tokenizer,
    )
    device = ppo_trainer.accelerator.device

    # TODO update training loop

    GLOBAL_ITER = 0
    
    



    # ppo_trainer.save_pretrained(f"{args.save_dir}/{run_name}-EPOCH-{0}") 
    # print(f"Model saved at {args.save_dir}/{run_name}-EPOCH-{0}")
                
                
    # LOG_EVERY = 10
    for epoch in tqdm(range(args.ppo_num_epochs)):
        for batch_idx, batch in tqdm(enumerate(ppo_trainer.dataloader)):
            
            print(f"Epoch {epoch} Batch {batch_idx}")
            
            generator_input_tokens = [ppo_trainer.tokenizer("You are", return_tensors='pt')['input_ids'].to(device).squeeze()] * ppo_config.batch_size
            
            print("GENERATOR INPUT TOKENS", generator_input_tokens[0], generator_input_tokens[0].shape)
            
            if args.broadcast:
                generator_output_tensors = [ppo_trainer.model.generate(input_ids=generator_input_tokens[0].unsqueeze(0), **generator_kwargs).squeeze()] * ppo_config.batch_size
            else:
                generator_output_tensors = [ppo_trainer.model.generate(input_ids=i.unsqueeze(0), **generator_kwargs).squeeze() for i in generator_input_tokens]
            
            batch["attack"] = ["".join(ppo_trainer.tokenizer.batch_decode(i)) for i in generator_output_tensors]
            target_inputs = [" ".join([attack, query]) for attack, query in zip(batch["attack"], batch["query"])]
            if GLOBAL_ITER % args.log_freq == 0:
                for i, (attack, query) in enumerate(zip(batch["attack"], batch["query"])):
                    if i < 3:
                        writer.add_text(f"target_input/target_input_attack[{i}]", attack, GLOBAL_ITER)
                        writer.add_text(f"target_input/target_input_query[{i}]", query, GLOBAL_ITER)
            
            target_outputs = [target.generate(i, target_kwargs) for i in target_inputs]
            
            if GLOBAL_ITER % args.log_freq == 0:
                for i, val in enumerate(target_outputs):
                    if i < 3:
                        writer.add_text(f"target_outputs/target_outputs[{i}]", val, GLOBAL_ITER)
            
            rewards = reward_model.generate(target_outputs)
            
            print("11111SAMPLE Target Input", 
                                        target_inputs[0], 
                                        "\n\n 11111SAMPLE ATTACK", 
                                        batch["attack"][0], 
                                        "\n\n 11111SAMPLE QUERY", 
                                        batch["query"][0], 
                                        "\n\n 11111TARGET_OUTPUT", 
                                        target_outputs[0])
            print("REWARDS_1", rewards[0])
            print("22222SAMPLE Target Input",
                                        target_inputs[1], 
                                        "\n\n 22222SAMPLE ATTACK", 
                                        batch["attack"][1], 
                                        "\n\n 22222SAMPLE QUERY", 
                                        batch["query"][1], 
                                        "\n\n 22222TARGET_OUTPUT", 
                                        target_outputs[1])
            print("REWARDS_2", rewards[1])
            if GLOBAL_ITER % args.log_freq == 0:
                writer.add_histogram("reward/rewards", rewards, GLOBAL_ITER)
                for i, val in enumerate(rewards):
                    if i < 3:
                        writer.add_scalar(f"reward/rewards[{i}]", val, GLOBAL_ITER)
            
            
            # TODO: Add diversity metrics here

            #### Run PPO step
            if args.broadcast:
                rewards = torch.tensor(rewards, device=device)
                print("ALL REWARDS", rewards)
                print("MAX REWARD", torch.max(rewards))
                rewards = [torch.max(rewards)] * args.ppo_batch_size
                stats = ppo_trainer.step(generator_input_tokens, generator_output_tensors, rewards) # only one sample update, since all are repeated
            else:
                rewards = [torch.tensor([item], device=device) for item in rewards]
                stats = ppo_trainer.step(generator_input_tokens, generator_output_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)
    
            # if not os.path.exists(args.save_dir):
            #     os.makedirs(args.save_dir)
            # ppo_trainer.save_model(args.save_dir)
            # generator.model.push_to_hub("my-fine-tuned-model-ppo")
            if GLOBAL_ITER % args.save_freq == 0:
                # ppo_trainer.save_model(args.save_dir)
                # get current time
                ppo_trainer.save_pretrained(f"{args.save_dir}/{run_name}-EPOCH-{epoch}-ITER-{GLOBAL_ITER}-BATCHSIZE-{args.ppo_batch_size}") 
                print(f"Model saved at {args.save_dir}/{run_name}-EPOCH-{epoch}-ITER-{GLOBAL_ITER}-BATCHSIZE-{args.ppo_batch_size}")
            
            GLOBAL_ITER += 1
            
        writer.flush()
            

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    ########### Generator model parameters ##########
    parser.add_argument(
        "--generator-model", 
        default = "sft_results/gpt2-medium/checkpoint-6675", # default = "gpt2-medium",
        help = "Name of attack generator model.",
        choices=["gpt2-medium", 
                 "gpt2-large", 
                 "gpt2-xl", 
                 "lvwerra/gpt2-imdb", 
                 "/data/public_models/zephyr/zephyr-7b-beta", 
                 "sft_results/gpt2-medium/checkpoint-6675"]
    )
    parser.add_argument(
        "--generator-max-tokens",
        type = int,
        default = 150,
        help = "Maximum number of generated tokens for the attacker."
    )

    parser.add_argument(
        "--generator-min-tokens",
        type = int,
        default = 10,
        help = "Minimum number of generated tokens for the attacker."
    )
    
    parser.add_argument(
        "--experiment-name",
        type = str,
        default = "PPO-baseline",
        help = "Experiment Name"
    )
    ##################################################

    ########### Target model parameters ##########
    parser.add_argument(
        "--target-model",
        default = "/data/public_models/zephyr/zephyr-7b-beta", # HuggingFaceH4/zephyr-7b-beta
        help = "Name of target model.",
        choices=["gpt2-medium", 
                 "lvwerra/gpt2-imdb", 
                 "/data/public_models/zephyr/zephyr-7b-beta", 
                 "/data/public_models/llama_v2_chat/Llama-2-7b-chat-hf",
                 "/data/public_models/vicuna/vicuna-7b-v1.3",
                 "/data/public_models/yi/Yi-34B"]
    )
    parser.add_argument(
        "--target-max-tokens",
        type = int,
        default = 150,
        help = "Maximum number of generated tokens for the target."
    )

    parser.add_argument(
        "--target-min-tokens",
        type = int,
        default = 10,
        help = "Minimum number of generated tokens for the target."
    )
    ##################################################

    # TODO: Add reward parameters
    ########### Reward model parameters ##########
    parser.add_argument(
        "--reward-model",
        default = "nicholasKluge/ToxicityModel",
        help = "Name of reward model.",
        choices=["nicholasKluge/ToxicityModel"]
    )
    parser.add_argument(
        "--reward-max-tokens",
        type = int,
        default = 150,
        help = "Maximum number of input tokens for the reward."
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
    
    ##################################################

    ########### Dataset parameters ##########
    parser.add_argument(
        "--dataset",
        default="./datasets/ppo_dataset_punctuation.csv",
        help = "Path to PPO dataset.",
        type = pathlib.Path,
    )
    
    ##################################################

    ########### PPO parameters ##########
    parser.add_argument(
        "--ppo-lr",
        default = 5e-5,
        help = "Learning rate for PPO",
        type = float,
    )

    parser.add_argument(
        "--ppo-num-epochs",
        default = 100,
        help = "Number of epochs for PPO",
        type = int,
    )

    parser.add_argument(
        "--ppo-batch-size",
        default = 20,
        help = "Batch size for PPO",
        type = int,
    )
    
    # TODO: add other PPO params
    ##################################################
    
    ########### Logging parameters ##########
    
    parser.add_argument(
        "--save-dir",
        default="./results",
        help = "Path to save the model.",
        type = pathlib.Path,
    )

    parser.add_argument(
        "--log-dir",
        default="./logs",
        help = "Path to save the logs.",
        type = pathlib.Path,
    )
    
    parser.add_argument(
        "--log-freq",
        default=10,
        help = "Logging frequency.",
        type = int,
    )
    parser.add_argument(
        "--save-freq",
        default=25,
        help = "Saving frequency.",
        type = int,
    )

    ##################################################
    
    parser.add_argument(
        "-b",
        "--broadcast",
        action="store_true",
    )
    
    args = parser.parse_args()
    main(args)